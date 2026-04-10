"""
IsoQuant: Quaternion-based 4D block rotation for KV cache quantization.

Drop-in replacement for RotorQuantMSE that uses quaternion sandwich products
instead of Clifford Cl(3,0) rotors. Based on the isoclinic decomposition:
    SO(4) ≅ SU(2) × SU(2)
    T(v) = q_L v q̄_R  (IsoQuant-Full, 6 DOF)
    T(v) = q_L v       (IsoQuant-Fast, 3 DOF)

Advantages over RotorQuant:
  - 4D blocks align to powers-of-2 head dims (no tail handling)
  - 4 components per block vs 8 (no multivector expansion)
  - ~2x fewer FMAs per rotation at same quality
  - Cleaner kernel fusion (4-wide = SIMD friendly)

Reference: Ji, "IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations
           for LLM KV Cache Compression", March 2026.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict

from .lloyd_max import LloydMaxCodebook


# ── Quaternion math ───────────────────────────────────────────────────

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate: (w, x, y, z) → (w, -x, -y, -z)."""
    signs = torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)
    return q * signs


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions.

    a, b: (..., 4) as [w, x, y, z]
    Returns: (..., 4) as [w, x, y, z]

    Uses 16 multiplies + 12 adds = 16 FMAs.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)

    rw = aw*bw - ax*bx - ay*by - az*bz
    rx = aw*bx + ax*bw + ay*bz - az*by
    ry = aw*by - ax*bz + ay*bw + az*bx
    rz = aw*bz + ax*by - ay*bx + az*bw

    return torch.stack([rw, rx, ry, rz], dim=-1)


def make_random_unit_quaternion(shape: Tuple[int, ...],
                                 device='cpu', seed=None) -> torch.Tensor:
    """Generate random unit quaternions via normalized Gaussian."""
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)
    q = torch.randn(*shape, 4, generator=gen).to(device)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ── IsoQuantMSE ───────────────────────────────────────────────────────

class IsoQuantMSE(nn.Module):
    """
    Stage 1: MSE-optimal quantizer using quaternion block rotations.

    Instead of Cl(3,0) rotor sandwich (3D blocks, 8 components),
    we use quaternion sandwich (4D blocks, 4 components).

    Modes:
      'full': T(v) = q_L v q̄_R  — full SO(4), 6 DOF per block
      'fast': T(v) = q_L v      — isoclinic SO(3) subgroup, 3 DOF per block
    """

    def __init__(self, d: int, bits: int, seed: int = 42,
                 mode: str = 'full', device: str = "cpu"):
        """
        Args:
            d: original vector dimension
            bits: bits per component for Lloyd-Max quantization
            seed: random seed for quaternion generation
            mode: 'full' (q_L v q̄_R) or 'fast' (q_L v)
            device: torch device
        """
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device
        self.mode = mode

        # 4D blocks — clean alignment for powers-of-2 dims
        self.n_groups = (d + 3) // 4  # ceil(d/4)
        self.d_padded = self.n_groups * 4

        # Lloyd-Max codebook: same as RotorQuant, d_eff = d
        cb = LloydMaxCodebook(d, bits)
        self.register_buffer('centroids', cb.centroids.to(device))

        # Random unit quaternions (one per group)
        q_L = make_random_unit_quaternion((self.n_groups,), device=device, seed=seed)
        self.register_buffer('q_L', q_L)

        if mode == 'full':
            q_R = make_random_unit_quaternion((self.n_groups,), device=device, seed=seed + 10000)
            self.register_buffer('q_R', q_R)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed d-dim vectors into groups of 4D quaternions.

        x: (..., d) → (..., n_groups, 4)
        """
        # Pad to multiple of 4
        pad = self.d_padded - self.d
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:-1], self.n_groups, 4)

    def _extract(self, v: torch.Tensor) -> torch.Tensor:
        """Extract d-dim vectors from groups of 4D quaternions.

        v: (..., n_groups, 4) → (..., d)
        """
        flat = v.reshape(*v.shape[:-2], -1)
        return flat[..., :self.d]

    def _rotate(self, v: torch.Tensor) -> torch.Tensor:
        """Apply forward rotation to 4D blocks.

        v: (..., n_groups, 4) — treated as quaternions
        """
        if self.mode == 'full':
            # T(v) = q_L * v * conjugate(q_R)
            temp = quat_multiply(self.q_L, v)
            return quat_multiply(temp, quat_conjugate(self.q_R))
        else:
            # T(v) = q_L * v
            return quat_multiply(self.q_L, v)

    def _unrotate(self, v: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation to 4D blocks.

        v: (..., n_groups, 4)
        """
        if self.mode == 'full':
            # T^{-1}(v) = conjugate(q_L) * v * q_R
            temp = quat_multiply(quat_conjugate(self.q_L), v)
            return quat_multiply(temp, self.q_R)
        else:
            # T^{-1}(v) = conjugate(q_L) * v
            return quat_multiply(quat_conjugate(self.q_L), v)

    def _quantize_scalar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-coordinate Lloyd-Max quantization."""
        diffs = x.unsqueeze(-1) - self.centroids  # (..., n_levels)
        indices = diffs.abs().argmin(dim=-1)
        x_q = self.centroids[indices]
        return x_q, indices

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize vectors via quaternion rotation + Lloyd-Max.

        x: (..., d) input vectors
        Returns: (x_quantized_rotated, indices_dict)
        """
        # Separate norm and direction
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # Embed into 4D blocks
        v = self._embed(x_unit)  # (..., n_groups, 4)

        # Rotate
        v_rot = self._rotate(v)

        # Scalar quantize all 4 components per block
        flat = v_rot.reshape(*v_rot.shape[:-2], -1)  # (..., n_groups*4)
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

        # Reshape back to groups of 4
        v_q = values.reshape(*values.shape[:-1], self.n_groups, 4)

        # Inverse rotation
        v_recon = self._unrotate(v_q)

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


# ── IsoQuantProd (Stage 1 + Stage 2 with QJL) ────────────────────────

class IsoQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.

    Uses (b-1)-bit IsoQuant MSE + 1-bit QJL on residuals.
    Drop-in replacement for RotorQuantProd.
    """

    def __init__(self, d: int, bits: int, mode: str = 'full',
                 qjl_dim: Optional[int] = None,
                 seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        # Stage 1: IsoQuant MSE
        self.mse = IsoQuantMSE(d, self.mse_bits, seed=seed, mode=mode, device=device)

        # Stage 2: QJL projection (same as TurboQuant/RotorQuant)
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
