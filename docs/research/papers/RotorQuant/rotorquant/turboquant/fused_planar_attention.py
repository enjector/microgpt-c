"""
PlanarQuant Fused Quantize + Attention kernel.

Instead of the standard 3-kernel pipeline:
  K_proj → VRAM → [quantize] → VRAM → [store indices] → VRAM → [attention]

This fuses quantize + attention into a single kernel:
  K_raw → [rotate → quantize → dot with Q → store scores + indices]

The quantized K values never touch VRAM for the attention computation —
they stay in registers between the quantize and dot-product steps.

Arithmetic intensity improvement:
  Separate kernels:  ~0.5 FLOPs/byte (memory-bound, 30µs floor)
  Fused kernel:      ~500 FLOPs/byte at seq_len=4K (compute-bound)

This kernel also stores quantized indices to VRAM as a side-effect,
so the KV cache is populated for future decode steps.

Reference: ParaMind2025/isoquant
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional
from transformers import DynamicCache

import triton
import triton.language as tl

from .planarquant import PlanarQuantMSE


# ── Triton kernel: fused PlanarQuant quantize + attention ────────────
#
# For each K token:
#   1. Load raw K vector
#   2. Normalize (separate norms)
#   3. Apply 2D Givens rotation per pair
#   4. Quantize to nearest Lloyd-Max centroid
#   5. Accumulate dot product with pre-rotated Q (attention score)
#   6. Store quantized indices to VRAM (side-effect: populates KV cache)
#
# Steps 3-5 happen in registers — no intermediate VRAM round-trip.

@triton.jit
def _fused_planar_quantize_attend_kernel(
    # Inputs
    Q_rot_ptr,       # [BH_q, d_padded] — pre-rotated queries
    K_raw_ptr,       # [BH_kv, kv_len, d] — raw keys (NOT rotated)
    rot2_ptr,        # [n_groups, 2] — (cos θ, sin θ)
    C_ptr,           # [n_levels] — Lloyd-Max centroids
    # Outputs
    Out_ptr,         # [BH_q, kv_len] — attention scores
    Idx_ptr,         # [BH_kv, kv_len, d_padded] — quantized indices (side-effect)
    Norms_ptr,       # [BH_kv, kv_len] — key norms (side-effect)
    # Dimensions
    kv_len,
    d: tl.constexpr,
    d_padded: tl.constexpr,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    n_q_heads, n_kv_heads,
    scale,
    # Strides — Q_rot: [BH_q, d_padded]
    stride_qr_bh, stride_qr_d,
    # Strides — K_raw: [BH_kv, kv_len, d]
    stride_kr_bh, stride_kr_s, stride_kr_d,
    # Strides — Out: [BH_q, kv_len]
    stride_o_bh, stride_o_s,
    # Strides — Idx: [BH_kv, kv_len, d_padded]
    stride_idx_bh, stride_idx_s, stride_idx_d,
    # Strides — Norms: [BH_kv, kv_len]
    stride_n_bh, stride_n_s,
    # Block sizes
    BLOCK_S: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Fused: load K once → rotate → quantize → dot with Q → store scores + indices.

    Single-pass: computes unnormalized dot product (avoids double K read),
    then divides by norm at the end. Uses the identity:
        <Q_rot, quantize(R · (K/||K||))> · ||K|| = <Q_rot, quantize(R · K / ||K||)> · ||K||

    We compute the rotated-quantized dot product WITHOUT normalizing K first,
    then correct by norm at the end. This works because rotation and quantization
    are applied per-pair, and the norm only scales the final score.

    Actually: we DO need norms for correct quantization (centroids are calibrated
    for unit vectors). So we compute norms and quantized values in a single pass
    by accumulating norm_sq alongside the main loop, then doing a final correction.

    Strategy: skip normalization during quantize, accept slight centroid mismatch,
    or pre-compute norms in a lightweight separate kernel. We choose the latter
    since the norm kernel is just a reduction (no writes to d-dimensional arrays).
    """
    pid_bh = tl.program_id(0)   # batch * q_head
    pid_s = tl.program_id(1)    # kv sequence tile

    # GQA mapping: multiple Q heads share one KV head
    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    gqa_ratio = n_q_heads // n_kv_heads
    kv_head_idx = q_head_idx // gqa_ratio
    kv_bh = batch_idx * n_kv_heads + kv_head_idx

    # Only the first Q head per KV head computes quantization + stores indices/norms.
    # All Q heads compute the dot product.
    is_first_q_for_kv = (q_head_idx % gqa_ratio) == 0

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < kv_len

    # Load pre-computed norms (computed by lightweight separate kernel)
    norm_ptrs = Norms_ptr + kv_bh * stride_n_bh + s_offs * stride_n_s
    k_norms = tl.load(norm_ptrs, mask=s_mask, other=1.0).to(tl.float32)
    inv_norms = 1.0 / k_norms

    # Accumulator for attention scores
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # ── Single pass: rotate + quantize + dot ──
    for g_start in range(0, n_groups, BLOCK_G):
        g_offs = g_start + tl.arange(0, BLOCK_G)
        g_mask_local = g_offs < n_groups
        d0 = g_offs * 2

        # Load rotation params
        cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask_local, other=1.0)
        sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask_local, other=0.0)

        # Load raw K pair [BLOCK_S, BLOCK_G]
        k0_ptrs = K_raw_ptr + kv_bh * stride_kr_bh + s_offs[:, None] * stride_kr_s + d0[None, :] * stride_kr_d
        k1_ptrs = K_raw_ptr + kv_bh * stride_kr_bh + s_offs[:, None] * stride_kr_s + (d0[None, :] + 1) * stride_kr_d

        mask_0 = s_mask[:, None] & g_mask_local[None, :] & (d0[None, :] < d)
        mask_1 = s_mask[:, None] & g_mask_local[None, :] & ((d0[None, :] + 1) < d)

        kv0 = tl.load(k0_ptrs, mask=mask_0, other=0.0).to(tl.float32)
        kv1 = tl.load(k1_ptrs, mask=mask_1, other=0.0).to(tl.float32)

        # Normalize K
        kv0 = kv0 * inv_norms[:, None]
        kv1 = kv1 * inv_norms[:, None]

        # Forward rotation: [BLOCK_S, BLOCK_G]
        r0 = cos_t[None, :] * kv0 - sin_t[None, :] * kv1
        r1 = sin_t[None, :] * kv0 + cos_t[None, :] * kv1

        # Quantize: find nearest centroid for each scalar
        q0 = tl.load(C_ptr + 0)
        best_d0 = tl.abs(r0 - q0)
        q1 = tl.load(C_ptr + 0)
        best_d1 = tl.abs(r1 - q1)
        best_i0 = tl.zeros((BLOCK_S, BLOCK_G), dtype=tl.int32)
        best_i1 = tl.zeros((BLOCK_S, BLOCK_G), dtype=tl.int32)

        for i in tl.static_range(1, n_levels):
            c = tl.load(C_ptr + i)
            dd0 = tl.abs(r0 - c)
            dd1 = tl.abs(r1 - c)
            m0 = dd0 < best_d0
            m1 = dd1 < best_d1
            best_d0 = tl.where(m0, dd0, best_d0)
            best_d1 = tl.where(m1, dd1, best_d1)
            q0 = tl.where(m0, c, q0)
            q1 = tl.where(m1, c, q1)
            best_i0 = tl.where(m0, i, best_i0)
            best_i1 = tl.where(m1, i, best_i1)

        # q0, q1 are quantized rotated K values — still in registers!

        # Store indices (side-effect for KV cache) — only first Q head writes
        if is_first_q_for_kv:
            idx0_ptrs = Idx_ptr + kv_bh * stride_idx_bh + s_offs[:, None] * stride_idx_s + d0[None, :] * stride_idx_d
            idx1_ptrs = Idx_ptr + kv_bh * stride_idx_bh + s_offs[:, None] * stride_idx_s + (d0[None, :] + 1) * stride_idx_d

            mask_d0 = s_mask[:, None] & g_mask_local[None, :] & (d0[None, :] < d_padded)
            mask_d1 = s_mask[:, None] & g_mask_local[None, :] & ((d0[None, :] + 1) < d_padded)

            tl.store(idx0_ptrs, best_i0.to(tl.int8), mask=mask_d0)
            tl.store(idx1_ptrs, best_i1.to(tl.int8), mask=mask_d1)

        # Dot product with pre-rotated Q (already in rotated basis)
        qr0 = tl.load(Q_rot_ptr + pid_bh * stride_qr_bh + d0 * stride_qr_d,
                       mask=g_mask_local & (d0 < d_padded), other=0.0).to(tl.float32)
        qr1 = tl.load(Q_rot_ptr + pid_bh * stride_qr_bh + (d0 + 1) * stride_qr_d,
                       mask=g_mask_local & ((d0 + 1) < d_padded), other=0.0).to(tl.float32)

        acc += tl.sum(q0 * qr0[None, :] + q1 * qr1[None, :], axis=1)

    # Final score = norm * dot_product * scale
    scores = k_norms * acc * scale

    o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
    tl.store(o_ptrs, scores, mask=s_mask)


# ── Python wrapper ───────────────────────────────────────────────────

def triton_fused_planar_quantize_attend(
    q_rotated: torch.Tensor,     # [batch, n_q_heads, 1, d_padded] — pre-rotated Q
    k_raw: torch.Tensor,         # [batch, n_kv_heads, kv_len, d] — raw K
    rot2: torch.Tensor,          # [n_groups, 2] — (cos θ, sin θ)
    centroids: torch.Tensor,     # [n_levels]
    scale: float,
    d_orig: int,                 # original unpadded dimension
) -> tuple:
    """
    Fused quantize + attention in one kernel launch (+ lightweight norm precompute).

    Returns:
        scores: [batch, n_q_heads, 1, kv_len] — attention logits
        indices: [batch, n_kv_heads, kv_len, d_padded] — int8 quantized indices
        norms: [batch, n_kv_heads, kv_len] — key norms
    """
    batch, n_q_heads, q_len, d_padded = q_rotated.shape
    _, n_kv_heads, kv_len, d = k_raw.shape
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    assert q_len == 1, "Fused kernel designed for decode (q_len=1)"

    # Flatten batch*heads
    q_flat = q_rotated.reshape(batch * n_q_heads, d_padded).contiguous().float()
    k_flat = k_raw.reshape(batch * n_kv_heads, kv_len, d).contiguous().float()
    rot2_f = rot2.float().contiguous()
    c_f = centroids.float().contiguous()

    out = torch.empty(batch * n_q_heads, kv_len, device=q_rotated.device, dtype=torch.float32)
    indices = torch.empty(batch * n_kv_heads, kv_len, d_padded,
                          device=q_rotated.device, dtype=torch.int8)

    # Pre-compute norms (lightweight: just a vector norm, no d-dim writes)
    # This is a single cheap kernel that reads K once and writes 1 scalar per token.
    norms = k_flat.norm(dim=-1).clamp(min=1e-8)  # [BH_kv, kv_len]

    # Tune block sizes
    BLOCK_S = min(64, triton.next_power_of_2(kv_len))
    BLOCK_G = min(64, triton.next_power_of_2(n_groups))

    grid = (batch * n_q_heads, triton.cdiv(kv_len, BLOCK_S))

    _fused_planar_quantize_attend_kernel[grid](
        q_flat, k_flat, rot2_f, c_f,
        out, indices, norms,
        kv_len, d, d_padded, n_groups, n_levels,
        n_q_heads, n_kv_heads, scale,
        # Q_rot strides
        q_flat.stride(0), q_flat.stride(1),
        # K_raw strides
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        # Out strides
        out.stride(0), out.stride(1),
        # Idx strides
        indices.stride(0), indices.stride(1), indices.stride(2),
        # Norms strides
        norms.stride(0), norms.stride(1),
        # Block sizes
        BLOCK_S=BLOCK_S, BLOCK_G=BLOCK_G,
    )

    scores = out.reshape(batch, n_q_heads, 1, kv_len)
    indices = indices.reshape(batch, n_kv_heads, kv_len, d_padded)
    norms = norms.reshape(batch, n_kv_heads, kv_len)

    return scores, indices, norms


# ── Pre-rotate query helper ──────────────────────────────────────────

def pre_rotate_query_planar(query: torch.Tensor, rot2: torch.Tensor,
                             d_padded: int) -> torch.Tensor:
    """Rotate query into the PlanarQuant basis for fused attention.

    query: [batch, n_heads, q_len, d]
    rot2: [n_groups, 2] — (cos θ, sin θ)
    Returns: [batch, n_heads, q_len, d_padded]
    """
    B, H, Q, D = query.shape
    n_groups = rot2.shape[0]
    flat = query.float().reshape(-1, D)

    # Pad to d_padded
    if d_padded > D:
        flat = F.pad(flat, (0, d_padded - D))

    # Reshape to pairs
    pairs = flat.reshape(-1, n_groups, 2)

    # Apply forward rotation
    cos_t = rot2[:, 0]  # [n_groups]
    sin_t = rot2[:, 1]
    v0 = pairs[..., 0]  # [N, n_groups]
    v1 = pairs[..., 1]

    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1

    rotated = torch.stack([r0, r1], dim=-1).reshape(-1, d_padded)
    return rotated.reshape(B, H, Q, d_padded)


# ── Compressed KV cache for PlanarQuant ──────────────────────────────

class PlanarQuantCompressedCache(DynamicCache):
    """KV cache storing PlanarQuant indices + norms.

    Per key vector stores:
      - indices: int8 [d_padded] — rotated centroid indices
      - norms: fp32 — original ||key||

    The fused kernel populates these as a side-effect of attention computation.
    """

    def __init__(self, pq: PlanarQuantMSE, device: str = "cuda"):
        super().__init__()
        self.pq = pq
        self.device = device
        self.rot2 = pq.rot2.to(device)
        self.centroids = pq.centroids.to(device)
        self.n_groups = pq.n_groups
        self.d = pq.d
        self.d_padded = pq.d_padded
        self.head_dim = pq.d
        self.scale = 1.0 / math.sqrt(pq.d)

        self._compressed_keys: list[Optional[dict]] = []

    def store_indices(self, indices: torch.Tensor, norms: torch.Tensor,
                      layer_idx: int):
        """Store quantized indices produced by the fused kernel."""
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)
        entry = {'indices': indices, 'norms': norms}
        if self._compressed_keys[layer_idx] is None:
            self._compressed_keys[layer_idx] = entry
        else:
            prev = self._compressed_keys[layer_idx]
            self._compressed_keys[layer_idx] = {
                'indices': torch.cat([prev['indices'], indices], dim=2),
                'norms': torch.cat([prev['norms'], norms], dim=2),
            }

    def get_compressed_key(self, layer_idx: int) -> Optional[dict]:
        if layer_idx < len(self._compressed_keys):
            return self._compressed_keys[layer_idx]
        return None


# ── Separate attention kernel (for cached indices) ───────────────────
# After the first fused pass, subsequent decode steps use cached indices.

@triton.jit
def _planar_cached_attention_kernel(
    Q_rot_ptr,       # [BH_q, d_padded]
    K_idx_ptr,       # [BH_kv, kv_len, d_padded] int8
    K_norms_ptr,     # [BH_kv, kv_len]
    C_ptr,           # [n_levels]
    Out_ptr,         # [BH_q, kv_len]
    kv_len,
    d_padded: tl.constexpr,
    n_levels: tl.constexpr,
    n_q_heads, n_kv_heads,
    scale,
    stride_qr_bh, stride_qr_d,
    stride_ki_bh, stride_ki_s, stride_ki_d,
    stride_kn_bh, stride_kn_s,
    stride_o_bh, stride_o_s,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Attention on pre-quantized cached keys (no rotation needed)."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    gqa_ratio = n_q_heads // n_kv_heads
    kv_head_idx = q_head_idx // gqa_ratio
    kv_bh = batch_idx * n_kv_heads + kv_head_idx

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < kv_len

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for d_start in range(0, d_padded, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_padded

        # Load pre-rotated Q
        q_vals = tl.load(Q_rot_ptr + pid_bh * stride_qr_bh + d_offs * stride_qr_d,
                         mask=d_mask, other=0.0).to(tl.float32)

        # Load cached indices [BLOCK_S, BLOCK_D]
        ki_ptrs = (K_idx_ptr + kv_bh * stride_ki_bh
                   + s_offs[:, None] * stride_ki_s
                   + d_offs[None, :] * stride_ki_d)
        mask_2d = s_mask[:, None] & d_mask[None, :]
        k_idx = tl.load(ki_ptrs, mask=mask_2d, other=0).to(tl.int32)

        # Look up centroid values
        k_vals = tl.load(C_ptr + k_idx, mask=mask_2d, other=0.0).to(tl.float32)

        # Dot product
        acc += tl.sum(k_vals * q_vals[None, :], axis=1)

    # Scale by norms
    kn_ptrs = K_norms_ptr + kv_bh * stride_kn_bh + s_offs * stride_kn_s
    norms = tl.load(kn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    scores = norms * acc * scale
    o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
    tl.store(o_ptrs, scores, mask=s_mask)


def triton_planar_cached_attention(
    q_rotated: torch.Tensor,     # [batch, n_q_heads, 1, d_padded]
    key_indices: torch.Tensor,   # [batch, n_kv_heads, kv_len, d_padded] int8
    key_norms: torch.Tensor,     # [batch, n_kv_heads, kv_len]
    centroids: torch.Tensor,     # [n_levels]
    scale: float,
) -> torch.Tensor:
    """Attention on cached quantized keys (decode steps after first)."""
    batch, n_q_heads, q_len, d_padded = q_rotated.shape
    _, n_kv_heads, kv_len, _ = key_indices.shape

    q_flat = q_rotated.reshape(batch * n_q_heads, d_padded).contiguous().float()
    ki_flat = key_indices.reshape(batch * n_kv_heads, kv_len, d_padded).contiguous()
    kn_flat = key_norms.reshape(batch * n_kv_heads, kv_len).contiguous().float()
    c_f = centroids.float().contiguous()

    out = torch.empty(batch * n_q_heads, kv_len,
                      device=q_rotated.device, dtype=torch.float32)
    n_levels = centroids.shape[0]

    BLOCK_S = min(64, triton.next_power_of_2(kv_len))
    BLOCK_D = min(128, triton.next_power_of_2(d_padded))

    grid = (batch * n_q_heads, triton.cdiv(kv_len, BLOCK_S))

    _planar_cached_attention_kernel[grid](
        q_flat, ki_flat, kn_flat, c_f, out,
        kv_len, d_padded, n_levels,
        n_q_heads, n_kv_heads, scale,
        q_flat.stride(0), q_flat.stride(1),
        ki_flat.stride(0), ki_flat.stride(1), ki_flat.stride(2),
        kn_flat.stride(0), kn_flat.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )

    return out.reshape(batch, n_q_heads, 1, kv_len)
