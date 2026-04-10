"""
Benchmark: PlanarQuant cached attention vs standard dequantize+matmul.

Simulates realistic autoregressive decode:
  - KV cache already populated with quantized indices (int8)
  - Each step: one new Q token attends to all cached K tokens
  - Compare three attention paths:
      A. Standard: dequantize K (inverse rotate) → VRAM → cuBLAS Q@K^T
      B. Cached:   gather-dot on int8 indices with pre-rotated Q (no dequantize)
      C. FP16 baseline: uncompressed K in VRAM → cuBLAS Q@K^T

Reference: ParaMind2025/isoquant
"""

import torch
import torch.nn.functional as F
import math
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from turboquant.planarquant import PlanarQuantMSE
from turboquant.triton_planarquant import triton_planar2_fused, triton_planar2_quantize
from turboquant.fused_planar_attention import (
    triton_planar_cached_attention,
    pre_rotate_query_planar,
)


def bench(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # µs


def benchmark_attention_paths(device='cuda'):
    """Compare three attention paths at different KV cache lengths."""
    d = 128
    bits = 3
    batch = 1
    n_q_heads = 32
    n_kv_heads = 8
    gqa_ratio = n_q_heads // n_kv_heads
    scale = 1.0 / math.sqrt(d)

    pq = PlanarQuantMSE(d, bits, seed=42, device=device)

    print("=" * 95)
    print("  DECODE ATTENTION: Standard (dequantize+matmul) vs Cached (gather-dot) vs FP16 baseline")
    print("=" * 95)
    print(f"  d={d}, bits={bits}, batch={batch}, q_heads={n_q_heads}, kv_heads={n_kv_heads}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print()
    print(f"  {'kv_len':>8} | {'FP16':>10} {'Standard':>10} {'Cached':>10} | {'Cached/FP16':>12} {'Cached/Std':>12} | {'Cos(C,S)':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*12}-{'-'*12}-+-{'-'*10}")

    for kv_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        torch.manual_seed(42)
        Q = torch.randn(batch, n_q_heads, 1, d, device=device)
        K_raw = torch.randn(batch, n_kv_heads, kv_len, d, device=device)

        # ── Pre-compute quantized cache (done once during prefill) ──
        K_flat = K_raw.reshape(-1, d)
        indices, norms = triton_planar2_quantize(K_flat, pq.rot2, pq.centroids)
        indices = indices.reshape(batch, n_kv_heads, kv_len, pq.d_padded)
        norms = norms.reshape(batch, n_kv_heads, kv_len)

        # Pre-rotate Q
        Q_rot = pre_rotate_query_planar(Q, pq.rot2, pq.d_padded)

        # Also pre-compute dequantized K for standard path
        K_deq = triton_planar2_fused(K_flat, pq.rot2, pq.centroids)
        K_deq = K_deq.reshape(batch, n_kv_heads, kv_len, d)

        # FP16 baseline (uncompressed K)
        K_fp16 = K_raw.half()
        K_fp16_exp = K_fp16.repeat_interleave(gqa_ratio, dim=1)
        Q_half = Q.half()

        # ── Path A: FP16 baseline ──
        def fp16_fn():
            return torch.matmul(Q_half, K_fp16_exp.transpose(2, 3)) * scale

        # ── Path B: Standard dequantize + matmul ──
        K_deq_exp = K_deq.repeat_interleave(gqa_ratio, dim=1)

        def standard_fn():
            return torch.matmul(Q, K_deq_exp.transpose(2, 3)) * scale

        # ── Path C: Cached gather-dot ──
        def cached_fn():
            return triton_planar_cached_attention(Q_rot, indices, norms, pq.centroids, scale)

        # Verify correctness
        ref = standard_fn().squeeze(2)
        out_c = cached_fn().squeeze(2)
        cos = F.cosine_similarity(ref.reshape(1, -1), out_c.reshape(1, -1)).item()

        # Benchmark
        t_fp16 = bench(fp16_fn)
        t_std = bench(standard_fn)
        t_cached = bench(cached_fn)

        print(f"  {kv_len:>8} | {t_fp16:>8.0f}µs {t_std:>8.0f}µs {t_cached:>8.0f}µs |"
              f" {t_fp16/t_cached:>11.2f}x {t_std/t_cached:>11.2f}x | {cos:>10.6f}")

    print()


def benchmark_memory(device='cuda'):
    """Show VRAM usage: standard path vs cached path."""
    d = 128
    bits = 3
    n_kv_heads = 8

    pq = PlanarQuantMSE(d, bits, seed=42, device=device)

    print("=" * 75)
    print("  KV CACHE MEMORY: FP16 vs Quantized (per layer)")
    print("=" * 75)
    print(f"  {'kv_len':>8} | {'FP16 K':>10} {'Quant K':>10} {'Compression':>12} | {'Quant format'}")
    print(f"  {'-'*8}-+-{'-'*10}-{'-'*10}-{'-'*12}-+-{'-'*30}")

    for kv_len in [1024, 4096, 16384, 32768, 65536, 131072]:
        # FP16: n_kv_heads × kv_len × d × 2 bytes
        fp16_bytes = n_kv_heads * kv_len * d * 2

        # Quantized: indices (int8) + norms (fp32)
        # indices: n_kv_heads × kv_len × d_padded × 1 byte
        # norms: n_kv_heads × kv_len × 4 bytes
        idx_bytes = n_kv_heads * kv_len * pq.d_padded * 1
        norm_bytes = n_kv_heads * kv_len * 4
        quant_bytes = idx_bytes + norm_bytes

        ratio = fp16_bytes / quant_bytes

        def fmt(b):
            if b < 1024:
                return f"{b} B"
            elif b < 1024 * 1024:
                return f"{b/1024:.1f} KB"
            else:
                return f"{b/(1024*1024):.1f} MB"

        print(f"  {kv_len:>8} | {fmt(fp16_bytes):>10} {fmt(quant_bytes):>10} {ratio:>11.1f}x |"
              f" int8[{pq.d_padded}] + fp32 norm")

    print()


def benchmark_decode_step(device='cuda'):
    """Simulate realistic decode: 1 new token per step, growing cache."""
    d = 128
    bits = 3
    batch = 1
    n_q_heads = 32
    n_kv_heads = 8
    scale = 1.0 / math.sqrt(d)

    pq = PlanarQuantMSE(d, bits, seed=42, device=device)

    print("=" * 75)
    print("  FULL DECODE STEP: quantize new K + attend to entire cache")
    print("=" * 75)
    print(f"  Simulates one autoregressive decode step at each cache length.")
    print(f"  'Std' = dequantize all cached K + matmul (what happens without fusion)")
    print(f"  'Fused' = quantize new K + gather-dot on cached + new K")
    print()
    print(f"  {'cache_len':>10} | {'Std total':>10} {'Fused total':>12} {'Speedup':>8}")
    print(f"  {'-'*10}-+-{'-'*10}-{'-'*12}-{'-'*8}")

    for cache_len in [256, 1024, 4096, 8192, 16384]:
        torch.manual_seed(42)
        Q = torch.randn(batch, n_q_heads, 1, d, device=device)

        # Cached K (already quantized from previous steps)
        K_cached_raw = torch.randn(batch, n_kv_heads, cache_len, d, device=device)
        K_flat = K_cached_raw.reshape(-1, d)
        cached_idx, cached_norms = triton_planar2_quantize(K_flat, pq.rot2, pq.centroids)
        cached_idx = cached_idx.reshape(batch, n_kv_heads, cache_len, pq.d_padded)
        cached_norms = cached_norms.reshape(batch, n_kv_heads, cache_len)

        # New K token (this step)
        K_new = torch.randn(batch, n_kv_heads, 1, d, device=device)

        # Pre-rotate Q
        Q_rot = pre_rotate_query_planar(Q, pq.rot2, pq.d_padded)

        # Dequantized cache for standard path
        K_deq = triton_planar2_fused(K_flat, pq.rot2, pq.centroids)
        K_deq = K_deq.reshape(batch, n_kv_heads, cache_len, d)

        gqa = n_q_heads // n_kv_heads

        # ── Standard: dequantize new K + concat + matmul ──
        def standard_step():
            # Quantize new K (for storage)
            Kn_flat = K_new.reshape(-1, d)
            Kn_hat = triton_planar2_fused(Kn_flat, pq.rot2, pq.centroids)
            Kn_hat = Kn_hat.reshape(batch, n_kv_heads, 1, d)
            # Concat with dequantized cache
            K_all = torch.cat([K_deq, Kn_hat], dim=2)
            K_exp = K_all.repeat_interleave(gqa, dim=1)
            return torch.matmul(Q, K_exp.transpose(2, 3)) * scale

        # ── Fused: quantize new K, concat indices, gather-dot ──
        def fused_step():
            # Quantize new K
            Kn_flat = K_new.reshape(-1, d)
            new_idx, new_norms = triton_planar2_quantize(Kn_flat, pq.rot2, pq.centroids)
            new_idx = new_idx.reshape(batch, n_kv_heads, 1, pq.d_padded)
            new_norms = new_norms.reshape(batch, n_kv_heads, 1)
            # Concat indices
            all_idx = torch.cat([cached_idx, new_idx], dim=2)
            all_norms = torch.cat([cached_norms, new_norms], dim=2)
            # Gather-dot attention
            return triton_planar_cached_attention(Q_rot, all_idx, all_norms,
                                                   pq.centroids, scale)

        t_std = bench(standard_step)
        t_fused = bench(fused_step)

        print(f"  {cache_len:>10} | {t_std:>8.0f}µs {t_fused:>10.0f}µs {t_std/t_fused:>7.2f}x")

    print()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    print()
    print("╔═════════════════════════════════════════════════════════════════════════════╗")
    print("║  PlanarQuant Fused Attention Benchmark                                    ║")
    print("║  Cached gather-dot vs standard dequantize+matmul                          ║")
    print("╚═════════════════════════════════════════════════════════════════════════════╝")
    print()

    benchmark_attention_paths()
    benchmark_memory()
    benchmark_decode_step()

    print("=" * 75)
    print("  DONE")
    print("=" * 75)
