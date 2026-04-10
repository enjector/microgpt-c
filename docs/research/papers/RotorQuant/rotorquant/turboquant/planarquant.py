"""
PlanarQuant: 2D planar rotation (Givens rotation) for KV cache quantization.

Simplest member of the rotation-based quantization family:
    SO(2): T(v) = R(θ) v  — single angle per 2D pair

Compared to other rotation quantizers:
  - RotorQuant: 3D blocks, Cl(3,0) rotor sandwich (8 components, ~28 FMAs)
  - IsoQuant:   4D blocks, quaternion sandwich (4 components, ~16 FMAs)
  - PlanarQuant: 2D blocks, Givens rotation (2 components, 4 FMAs)

Advantages:
  - Minimum overhead per group: 1 angle (or cos/sin pair) per 2 elements
  - Trivially vectorizable — no quaternion algebra needed
  - Pairs align to any even dimension (most common in practice)
  - CUDA kernel fuses rotate→quantize→unrotate in one pass

Disadvantage:
  - Only 1 DOF per 2D pair — less decorrelation power than 4D quaternion

Reference CUDA kernel: isoquant/csrc/planar2_fused_kernel.cu
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .lloyd_max import LloydMaxCodebook


# ── 2D rotation math ────────────────────────────────────────────────

def make_random_rotations(n_groups: int, device='cpu', seed=None) -> torch.Tensor:
    """Generate random 2D rotation parameters (cos θ, sin θ) per group.

    Returns: (n_groups, 2) as [cos θ, sin θ]
    """
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)
    # Random angles in [0, 2π)
    angles = torch.rand(n_groups, generator=gen) * (2 * math.pi)
    angles = angles.to(device)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def rot2_apply(cs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotation to pairs.

    cs: (..., 2) as [cos θ, sin θ]
    v:  (..., 2) as [v0, v1]
    Returns: (..., 2)
    """
    c = cs[..., 0]
    s = cs[..., 1]
    v0 = v[..., 0]
    v1 = v[..., 1]
    return torch.stack([c * v0 - s * v1, s * v0 + c * v1], dim=-1)


def rot2_inverse(cs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply inverse 2D rotation (transpose = negate sin).

    cs: (..., 2) as [cos θ, sin θ]
    v:  (..., 2) as [v0, v1]
    Returns: (..., 2)
    """
    c = cs[..., 0]
    s = cs[..., 1]
    v0 = v[..., 0]
    v1 = v[..., 1]
    return torch.stack([c * v0 + s * v1, -s * v0 + c * v1], dim=-1)


# ── PlanarQuantMSE ───────────────────────────────────────────────────

class PlanarQuantMSE(nn.Module):
    """
    Stage 1: MSE-optimal quantizer using 2D planar (Givens) rotations.

    Each pair of adjacent elements is rotated by an independent angle θ,
    then scalar-quantized via Lloyd-Max, then inverse-rotated.

    This is the SO(2) analogue of IsoQuant's SO(4) quaternion rotations.
    """

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        """
        Args:
            d: original vector dimension
            bits: bits per component for Lloyd-Max quantization
            seed: random seed for rotation generation
            device: torch device
        """
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        # 2D blocks — pairs
        self.n_groups = (d + 1) // 2  # ceil(d/2)
        self.d_padded = self.n_groups * 2

        # Lloyd-Max codebook
        cb = LloydMaxCodebook(d, bits)
        self.register_buffer('centroids', cb.centroids.to(device))

        # Random 2D rotations (cos θ, sin θ) — one per group
        rot = make_random_rotations(self.n_groups, device=device, seed=seed)
        self.register_buffer('rot2', rot)  # (n_groups, 2)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed d-dim vectors into groups of 2D pairs.

        x: (..., d) → (..., n_groups, 2)
        """
        pad = self.d_padded - self.d
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:-1], self.n_groups, 2)

    def _extract(self, v: torch.Tensor) -> torch.Tensor:
        """Extract d-dim vectors from groups of 2D pairs.

        v: (..., n_groups, 2) → (..., d)
        """
        flat = v.reshape(*v.shape[:-2], -1)
        return flat[..., :self.d]

    def _quantize_scalar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-coordinate Lloyd-Max quantization."""
        diffs = x.unsqueeze(-1) - self.centroids  # (..., n_levels)
        indices = diffs.abs().argmin(dim=-1)
        x_q = self.centroids[indices]
        return x_q, indices

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize vectors via 2D rotation + Lloyd-Max.

        x: (..., d) input vectors
        Returns: (x_quantized_rotated, indices_dict)
        """
        # Separate norm and direction
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # Embed into 2D pairs
        v = self._embed(x_unit)  # (..., n_groups, 2)

        # Rotate each pair
        v_rot = rot2_apply(self.rot2, v)

        # Scalar quantize
        flat = v_rot.reshape(*v_rot.shape[:-2], -1)  # (..., n_groups*2)
        q_flat, indices = self._quantize_scalar(flat)
        v_q = q_flat.reshape_as(v_rot)

        return v_q, {
            'indices': indices,
            '_norms': norms.squeeze(-1),
        }

    def dequantize(self, indices_dict: dict) -> torch.Tensor:
        """Reconstruct vectors from quantized indices."""
        idx = indices_dict['indices']
        values = self.centroids[idx]

        # Reshape back to groups of 2
        v_q = values.reshape(*values.shape[:-1], self.n_groups, 2)

        # Inverse rotation
        v_recon = rot2_inverse(self.rot2, v_q)

        # Extract and rescale
        x_hat = self._extract(v_recon)
        if '_norms' in indices_dict:
            norms = indices_dict['_norms']
            if norms.dim() < x_hat.dim():
                norms = norms.unsqueeze(-1)
            x_hat = x_hat * norms

        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Full quantize-dequantize cycle."""
        v_q, indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


# ── PlanarQuantProd (Stage 1 + Stage 2 with QJL) ────────────────────

class PlanarQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.

    Uses (b-1)-bit PlanarQuant MSE + 1-bit QJL on residuals.
    Drop-in replacement for IsoQuantProd / RotorQuantProd.
    """

    def __init__(self, d: int, bits: int,
                 qjl_dim: Optional[int] = None,
                 seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        # Stage 1: PlanarQuant MSE
        self.mse = PlanarQuantMSE(d, self.mse_bits, seed=seed, device=device)

        # Stage 2: QJL projection
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed + 1)
        S = torch.randn(self.qjl_dim, d, generator=gen)
        self.register_buffer("S", S.to(device))

    def quantize(self, x: torch.Tensor) -> dict:
        x_hat, mse_indices = self.mse(x)

        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)

        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0

        return {
            'mse_indices': mse_indices,
            'qjl_signs': qjl_signs,
            'residual_norm': residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        return self.mse.dequantize(compressed['mse_indices'])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        x_mse = self.mse.dequantize(compressed['mse_indices'])
        term1 = (y * x_mse).sum(dim=-1)

        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed['qjl_signs']).sum(dim=-1)

        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed['residual_norm'] * correction_scale * qjl_ip

        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict:
        return self.quantize(x)
