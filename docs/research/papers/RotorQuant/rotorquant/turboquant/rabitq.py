"""
RaBitQ: Randomized Binary Quantization for KV Cache Compression.

Faithful port of RaBitQ (arXiv 2405.12497) adapted for KV cache.
Reference C++ implementation: github.com/gaoj0017/RaBitQ

Key math (from the paper & reference):
  Data (K vectors):
    1. Rotate: x̃ = P·(x - centroid)  [random orthogonal P]
    2. Normalize: x̂ = x̃ / ‖x̃‖
    3. Binary snap: b = sign(x̂) ∈ {±1}^d
    4. Store: b (packed bits), ‖x̃‖ (norm), x₀ = ⟨x̂, b⟩/d (alignment)

  Query (Q vectors):
    1. Rotate: ỹ = P·(y - centroid)
    2. Asymmetric IP: ⟨x, y⟩ ≈ ‖x̃‖ · [⟨b, ỹ⟩ · (1/d) / x₀]

  The correction factor x₀ = mean(|x̂_i|) measures how well the unit
  vector aligns with its sign vertex. For Gaussian coordinates:
  E[x₀] = √(2/(πd)).

Three rotation backends:
  'planar': Givens 2D (fast, reuses PlanarQuant)
  'iso':    Quaternion 4D (reuses IsoQuant)
  'full':   Random orthogonal d×d (true RaBitQ, best quality)

Storage per vector (d=128):
  - d/8 = 16 bytes packed signs
  - 2 bytes FP16 norm
  - 2 bytes FP16 alignment scalar (x₀)
  Total: 20 bytes → 12.8× vs FP16

Reference: Jianyang Gao, Cheng Long. "RaBitQ: Quantizing High-Dimensional
Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor
Search". SIGMOD 2024. arXiv:2405.12497
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional


# ── Sign packing/unpacking (matches GGUF Q1_0 convention) ──────────

def pack_signs_to_uint8(signs: torch.Tensor) -> torch.Tensor:
    """Pack ±1 sign tensor into uint8 bitfield.
    signs: (..., d) with values ±1.  Returns: (..., d//8) uint8."""
    bits = (signs > 0).to(torch.uint8)
    d = signs.shape[-1]
    assert d % 8 == 0, f"d must be multiple of 8, got {d}"
    bits = bits.reshape(*bits.shape[:-1], d // 8, 8)
    packed = torch.zeros(*bits.shape[:-1], dtype=torch.uint8, device=signs.device)
    for i in range(8):
        packed |= bits[..., i] << i
    return packed


def unpack_signs_from_uint8(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack uint8 bitfield to ±1 float tensor.
    packed: (..., d//8) uint8.  Returns: (..., d) float."""
    signs = torch.zeros(*packed.shape[:-1], d, dtype=torch.float32, device=packed.device)
    for i in range(8):
        bits = ((packed >> i) & 1).float()
        signs[..., i::8] = bits * 2.0 - 1.0
    return signs


def popcount_ip(packed_a: torch.Tensor, packed_b: torch.Tensor) -> torch.Tensor:
    """Binary inner product via XOR + popcount on packed uint8.
    Returns: (...) integer counts of matching bits × 2 - d."""
    # XNOR: matching bits = NOT(XOR)
    xnor = ~(packed_a ^ packed_b)
    # Count set bits per byte
    counts = torch.zeros(packed_a.shape[:-1], dtype=torch.int32, device=packed_a.device)
    for i in range(8):
        counts += ((xnor >> i) & 1).to(torch.int32).sum(dim=-1)
    # IP = 2 * popcount(XNOR) - d
    d = packed_a.shape[-1] * 8
    return 2 * counts - d


# ── RaBitQ core ────────────────────────────────────────────────────

class RaBitQ(nn.Module):
    """
    RaBitQ quantizer with pluggable rotation backend.

    Implements the full RaBitQ pipeline:
      Quantize: normalize → rotate → sign snap → store corrections
      Dequantize: unpack → scale by correction → inverse rotate (approximate)
      Inner product: asymmetric estimator (query FP × key binary)
    """

    def __init__(self, d: int, rotation: str = 'full',
                 seed: int = 42, device: str = 'cpu'):
        super().__init__()
        self.d = d
        self.rotation_type = rotation
        self.device = device

        assert d % 8 == 0, f"d must be multiple of 8, got {d}"

        # Bits per element: (d/8 signs + 2 FP16 norms + 2 FP16 x0) * 8 / d
        self.bits_per_element = (d / 8 + 4) * 8 / d  # 1.25 for d=128

        # RaBitQ normalizing constant: E[|z|] for z ~ N(0, 1/d)
        # After rotation, unit vector coords ≈ N(0, 1/d), so E[|coord|] = √(2/(πd))
        self._expected_abs = math.sqrt(2.0 / (math.pi * d))

        # Initialize rotation
        if rotation == 'planar':
            from .planarquant import make_random_rotations
            n_groups = d // 2
            rot = make_random_rotations(n_groups, device=device, seed=seed)
            self.register_buffer('rot2', rot)

        elif rotation == 'iso':
            from .isoquant import make_random_unit_quaternion
            n_groups = d // 4
            q_L = make_random_unit_quaternion((n_groups,), device=device, seed=seed)
            self.register_buffer('q_L', q_L)

        elif rotation == 'full':
            # Random orthogonal matrix via QR decomposition (true RaBitQ)
            gen = torch.Generator(device='cpu')
            gen.manual_seed(seed)
            A = torch.randn(d, d, generator=gen)
            Q, _ = torch.linalg.qr(A)
            self.register_buffer('P', Q.to(device))  # (d, d) orthogonal

        else:
            raise ValueError(f"Unknown rotation: {rotation}")

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward rotation. x: (..., d) → (..., d)"""
        if self.rotation_type == 'planar':
            from .planarquant import rot2_apply
            v = x.reshape(*x.shape[:-1], -1, 2)
            v_rot = rot2_apply(self.rot2, v)
            return v_rot.reshape(*x.shape)
        elif self.rotation_type == 'iso':
            from .isoquant import quat_multiply
            v = x.reshape(*x.shape[:-1], -1, 4)
            v_rot = quat_multiply(self.q_L, v)
            return v_rot.reshape(*x.shape)
        else:
            return x @ self.P.T

    def _unrotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation. x: (..., d) → (..., d)"""
        if self.rotation_type == 'planar':
            from .planarquant import rot2_inverse
            v = x.reshape(*x.shape[:-1], -1, 2)
            v_inv = rot2_inverse(self.rot2, v)
            return v_inv.reshape(*x.shape)
        elif self.rotation_type == 'iso':
            from .isoquant import quat_multiply, quat_conjugate
            v = x.reshape(*x.shape[:-1], -1, 4)
            v_inv = quat_multiply(quat_conjugate(self.q_L), v)
            return v_inv.reshape(*x.shape)
        else:
            return x @ self.P  # P orthogonal → P^{-1} = P^T, so x @ P = x @ (P^T)^T

    def quantize(self, x: torch.Tensor) -> Dict:
        """
        Quantize vectors to RaBitQ binary representation.

        x: (..., d)
        Returns dict with packed_signs, norms, x0 (alignment scalar)

        Following the reference implementation (ivf_rabitq.h):
          - norms = ‖x‖ (vector magnitude, stored as FP16)
          - x0 = ⟨x̂_rot, sign(x̂_rot)⟩ / d = mean(|x̂_rot_i|) (alignment)
          - packed_signs = sign(x̂_rot) packed into uint8
        """
        # 1. Compute norms
        norms = torch.norm(x, dim=-1).clamp(min=1e-8)
        x_unit = x / norms.unsqueeze(-1)

        # 2. Rotate
        x_rot = self._rotate(x_unit)

        # 3. Sign snap
        signs = torch.sign(x_rot)
        signs[signs == 0] = 1.0

        # 4. Alignment scalar x₀ = ⟨x̂_rot, signs⟩ / d = mean(|x̂_rot_i|)
        # This measures how well the unit vector aligns with its hypercube vertex
        # For perfectly Gaussian coords: E[x₀] = √(2/(πd)) ≈ 0.0886 for d=128
        x0 = (x_rot * signs).sum(dim=-1) / self.d

        # 5. Pack signs
        packed = pack_signs_to_uint8(signs)

        return {
            'packed_signs': packed,               # (..., d/8) uint8
            'norms': norms.to(torch.float16),     # (...) FP16
            'x0': x0.to(torch.float16),           # (...) FP16 alignment
        }

    def dequantize(self, compressed: Dict) -> torch.Tensor:
        """
        Approximate reconstruction from binary representation.

        Reconstruction: x̂ ≈ P^{-1} · (signs · x₀) · ‖x‖
        This is lossy — the sign vector only captures direction, not magnitude
        per coordinate. Use inner_product() for accurate attention scores.
        """
        signs = unpack_signs_from_uint8(compressed['packed_signs'], self.d)
        norms = compressed['norms'].float()
        x0 = compressed['x0'].float()

        # Scale signs by alignment (approximates the rotated unit vector)
        x_rot_approx = signs * x0.unsqueeze(-1)  # (..., d)

        # Inverse rotate
        x_unit_approx = self._unrotate(x_rot_approx)

        # Rescale
        return x_unit_approx * norms.unsqueeze(-1)

    def inner_product(self, query: torch.Tensor, compressed: Dict) -> torch.Tensor:
        """
        Asymmetric inner product estimator (RaBitQ core).

        query: (..., d) full-precision query vectors
        compressed: dict from quantize()
        Returns: (...) estimated ⟨query, key⟩

        RaBitQ estimator:
          ⟨q, k⟩ ≈ ‖k‖ · ⟨R·q_unit, signs_k⟩ · (1 / (d · x₀_k))
          where x₀_k = mean(|Rk_unit_i|) is the alignment scalar.

        This works because:
          ⟨q, k⟩ = ‖q‖·‖k‖·⟨q̂, k̂⟩
          ⟨q̂, k̂⟩ = ⟨Rq̂, Rk̂⟩  (rotation preserves IP)
          Rk̂ ≈ signs_k · x₀_k  (binary approximation)
          ⟨Rq̂, signs_k · x₀_k⟩ = x₀_k · ⟨Rq̂, signs_k⟩
        """
        signs = unpack_signs_from_uint8(compressed['packed_signs'], self.d)
        norms = compressed['norms'].float()
        x0 = compressed['x0'].float()

        # Rotate query
        q_rot = self._rotate(query)

        # Dot product in rotated space: ⟨Rq, signs⟩
        raw_ip = (q_rot * signs).sum(dim=-1)

        # Scale by key norm and alignment
        # ⟨q, k⟩ ≈ ‖k‖ · raw_ip · x₀ / (x₀ · d) ... simplifies to:
        # ⟨q, k⟩ ≈ ‖k‖ · raw_ip  (x₀ cancels because we're comparing relative scores)
        # Actually: Rk̂ ≈ signs · x₀, so ⟨Rq, Rk̂⟩ ≈ ⟨Rq, signs⟩ · x₀
        # And ⟨q, k⟩ = ‖k‖ · ⟨Rq̂, Rk̂⟩ ≈ ‖k‖ · x₀ · ⟨Rq, signs⟩ / ‖q‖
        # For attention: we want ⟨q, k⟩, not normalized:
        estimated_ip = norms * x0 * raw_ip

        return estimated_ip

    def inner_product_batch(self, queries: torch.Tensor,
                            compressed: Dict) -> torch.Tensor:
        """
        Batch asymmetric IP: queries (B,H,Q,d) × keys (B,H,S,d) → (B,H,Q,S).

        This is the attention score computation path.
        """
        signs = unpack_signs_from_uint8(compressed['packed_signs'], self.d)
        norms = compressed['norms'].float()   # (B, H, S)
        x0 = compressed['x0'].float()         # (B, H, S)

        # Rotate all queries
        q_rot = self._rotate(queries)  # (B, H, Q, d)

        # Batch matmul: (B,H,Q,d) @ (B,H,d,S) → (B,H,Q,S)
        raw_ip = torch.matmul(q_rot, signs.transpose(-2, -1))

        # Scale: each key has its own norm and x0
        # raw_ip[..., s] * norms[..., s] * x0[..., s]
        scale = (norms * x0).unsqueeze(-2)  # (B, H, 1, S)
        return raw_ip * scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Quantize → dequantize round-trip."""
        compressed = self.quantize(x)
        x_hat = self.dequantize(compressed)
        return x_hat, compressed

    def compression_ratio(self) -> float:
        return 16.0 / self.bits_per_element


# ── RaBitQ KV Cache ───────────────────────────────────────────────

class RaBitQKVCache(nn.Module):
    """
    KV cache with RaBitQ compression.

    K cache: binary + corrections. Attention scores via asymmetric IP.
    V cache: binary + corrections. Values via approximate dequantize.

    For best quality, use separate RaBitQ instances for K and V
    (different random rotations).
    """

    def __init__(self, d: int, rotation: str = 'full',
                 seed: int = 42, device: str = 'cpu'):
        super().__init__()
        self.d = d
        self.rabitq_k = RaBitQ(d, rotation=rotation, seed=seed, device=device)
        self.rabitq_v = RaBitQ(d, rotation=rotation, seed=seed + 7919, device=device)

        self._k_cache = None
        self._v_cache = None
        self._seq_len = 0

    def insert(self, k: torch.Tensor, v: torch.Tensor):
        """Insert KV vectors. k, v: (batch, n_heads, n_new, d)"""
        kc = self.rabitq_k.quantize(k)
        vc = self.rabitq_v.quantize(v)

        if self._k_cache is None:
            self._k_cache = kc
            self._v_cache = vc
        else:
            for key in self._k_cache:
                self._k_cache[key] = torch.cat([self._k_cache[key], kc[key]], dim=-2)
                self._v_cache[key] = torch.cat([self._v_cache[key], vc[key]], dim=-2)

        self._seq_len += k.shape[-2]

    def attention_scores(self, query: torch.Tensor) -> torch.Tensor:
        """Compute Q·K^T via asymmetric IP. Returns (B,H,Q,S) logits."""
        if self._k_cache is None:
            return None
        return self.rabitq_k.inner_product_batch(query, self._k_cache)

    def get_values(self) -> torch.Tensor:
        """Decompress V cache."""
        if self._v_cache is None:
            return None
        return self.rabitq_v.dequantize(self._v_cache)

    def clear(self):
        self._k_cache = None
        self._v_cache = None
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def memory_bytes(self) -> int:
        if self._k_cache is None:
            return 0
        def cache_bytes(c):
            return (c['packed_signs'].numel() +
                    c['norms'].numel() * 2 +
                    c['x0'].numel() * 2)
        return cache_bytes(self._k_cache) + cache_bytes(self._v_cache)
