"""
Cross-validation: RotorQuant variants vs TurboQuant+ (TheTom/turboquant_plus).

Compares our rotation-based quantizers against the reference dense-rotation
TurboQuant+ implementation on identical data and metrics:

  1. Reconstruction MSE (synthetic unit vectors)
  2. Inner product preservation (Stage 1 + Stage 2)
  3. Codebook agreement (Lloyd-Max centroids)
  4. Speed (quantize + dequantize roundtrip)
  5. Parameter count

TurboQuant+ uses numpy; our methods use PyTorch on GPU.
We ensure identical random vectors for fair comparison.

Usage:
    python -m turboquant.benchmark_vs_tqplus
"""

import sys
import os
import time
import math
import numpy as np
import torch

# ── Import our methods FIRST ──
from turboquant.rotorquant import RotorQuantMSE, RotorQuantProd
from turboquant.isoquant import IsoQuantMSE, IsoQuantProd
from turboquant.planarquant import PlanarQuantMSE, PlanarQuantProd
from turboquant.lloyd_max import LloydMaxCodebook

# ── Import TurboQuant+ (numpy-based) ──
# Both packages are named "turboquant" so we temporarily swap sys.modules
TQPLUS_DIR = '/home/johndpope/Documents/turboquant_plus'

# Save our turboquant modules, load TQ+ in isolated namespace
import importlib
_saved_mods = {k: v for k, v in sys.modules.items() if k.startswith('turboquant')}
for k in list(_saved_mods.keys()):
    del sys.modules[k]
sys.path.insert(0, TQPLUS_DIR)

from turboquant.polar_quant import PolarQuant as TQPlus_PolarQuant
from turboquant.turboquant import TurboQuant as TQPlus_TurboQuant, TurboQuantMSE as TQPlus_TurboQuantMSE
from turboquant.codebook import optimal_centroids as tqplus_centroids

# Restore our modules
for k in list(sys.modules.keys()):
    if k.startswith('turboquant'):
        del sys.modules[k]
sys.path.remove(TQPLUS_DIR)
sys.modules.update(_saved_mods)


def make_test_vectors(n, d, seed=42):
    """Generate identical test vectors for both frameworks."""
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((n, d)).astype(np.float32)
    # Normalize to unit sphere
    norms = np.linalg.norm(x_np, axis=1, keepdims=True)
    x_np = x_np / norms
    x_torch = torch.from_numpy(x_np)
    return x_np, x_torch


def benchmark_codebook(d, bits):
    """Compare Lloyd-Max codebooks between the two implementations."""
    # TQPlus
    c_tqplus = tqplus_centroids(bits, d)  # numpy array

    # Ours
    cb = LloydMaxCodebook(d, bits)
    c_ours = cb.centroids.numpy()

    # Compare
    max_diff = np.max(np.abs(np.sort(c_tqplus) - np.sort(c_ours)))
    return max_diff, c_tqplus, c_ours


def benchmark_mse(d, bits, n_vectors=8192, device='cuda'):
    """Compare reconstruction MSE across all methods."""
    x_np, x_torch = make_test_vectors(n_vectors, d, seed=42)
    x_gpu = x_torch.to(device)

    results = {}

    # ── TQPlus: PolarQuant (dense d×d rotation) ──
    pq_plus = TQPlus_PolarQuant(d, bit_width=bits, seed=42)
    indices_np, norms_np = pq_plus.quantize(x_np)
    x_hat_np = pq_plus.dequantize(indices_np, norms_np)
    mse_tqplus = float(np.mean((x_np - x_hat_np) ** 2))
    results['TQ+ PolarQuant'] = mse_tqplus

    # ── Our methods ──
    rq = RotorQuantMSE(d, bits, seed=42, device=device)
    x_hat_rq, _ = rq(x_gpu)
    mse_rq = (x_gpu - x_hat_rq).pow(2).mean().item()
    results['RotorQuant'] = mse_rq

    iq_full = IsoQuantMSE(d, bits, seed=42, mode='full', device=device)
    x_hat_iq, _ = iq_full(x_gpu)
    mse_iq = (x_gpu - x_hat_iq).pow(2).mean().item()
    results['IsoQuant-Full'] = mse_iq

    iq_fast = IsoQuantMSE(d, bits, seed=42, mode='fast', device=device)
    x_hat_if, _ = iq_fast(x_gpu)
    mse_if = (x_gpu - x_hat_if).pow(2).mean().item()
    results['IsoQuant-Fast'] = mse_if

    pq = PlanarQuantMSE(d, bits, seed=42, device=device)
    x_hat_pq, _ = pq(x_gpu)
    mse_pq = (x_gpu - x_hat_pq).pow(2).mean().item()
    results['PlanarQuant'] = mse_pq

    return results


def benchmark_inner_product(d, bits, n_vectors=4096, device='cuda'):
    """Compare inner product estimation (Stage 1 + Stage 2)."""
    x_np, x_torch = make_test_vectors(n_vectors, d, seed=42)
    y_np, y_torch = make_test_vectors(n_vectors, d, seed=123)
    x_gpu = x_torch.to(device)
    y_gpu = y_torch.to(device)

    # True inner products
    true_ip = np.sum(x_np * y_np, axis=1)

    results = {}

    # ── TQPlus: TurboQuant (PolarQuant + QJL) ──
    tq_plus = TQPlus_TurboQuant(d, bit_width=bits, seed=42)
    compressed_x = tq_plus.quantize(x_np)
    x_hat_tq = tq_plus.dequantize(compressed_x)
    est_ip_tq = np.sum(x_hat_tq * y_np, axis=1)
    results['TQ+ TurboQuant'] = {
        'ip_mse': float(np.mean((true_ip - est_ip_tq) ** 2)),
        'ip_bias': float(np.mean(true_ip - est_ip_tq)),
    }

    # ── Our methods ──
    methods_torch = {
        'RotorQuant': RotorQuantProd(d, bits, seed=42, device=device),
        'IsoQuant-Full': IsoQuantProd(d, bits, mode='full', seed=42, device=device),
        'IsoQuant-Fast': IsoQuantProd(d, bits, mode='fast', seed=42, device=device),
        'PlanarQuant': PlanarQuantProd(d, bits, seed=42, device=device),
    }

    true_ip_torch = (x_gpu * y_gpu).sum(dim=-1)

    for name, model in methods_torch.items():
        compressed = model.quantize(x_gpu)
        est_ip = model.inner_product(y_gpu, compressed)
        ip_mse = (true_ip_torch - est_ip).pow(2).mean().item()
        ip_bias = (true_ip_torch - est_ip).mean().item()
        results[name] = {'ip_mse': ip_mse, 'ip_bias': ip_bias}

    return results


def benchmark_speed(d, bits, n_vectors=8192, device='cuda'):
    """Compare quantize+dequantize latency."""
    x_np, x_torch = make_test_vectors(n_vectors, d, seed=42)
    x_gpu = x_torch.to(device)

    results = {}

    # ── TQPlus (numpy, CPU) ──
    pq_plus = TQPlus_PolarQuant(d, bit_width=bits, seed=42)
    # Warmup
    for _ in range(3):
        idx, norms = pq_plus.quantize(x_np)
        pq_plus.dequantize(idx, norms)
    t0 = time.perf_counter()
    for _ in range(20):
        idx, norms = pq_plus.quantize(x_np)
        pq_plus.dequantize(idx, norms)
    dt = (time.perf_counter() - t0) / 20 * 1e6
    results['TQ+ PolarQuant (CPU)'] = dt

    # ── Our methods (GPU) ──
    methods = {
        'RotorQuant': RotorQuantMSE(d, bits, seed=42, device=device),
        'IsoQuant-Fast': IsoQuantMSE(d, bits, seed=42, mode='fast', device=device),
        'PlanarQuant': PlanarQuantMSE(d, bits, seed=42, device=device),
    }

    for name, model in methods.items():
        for _ in range(50):
            model(x_gpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            model(x_gpu)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 200 * 1e6
        results[name] = dt

    return results


def benchmark_params(d, bits):
    """Compare parameter counts."""
    # TQPlus uses a d×d rotation matrix
    tqplus_rot_params = d * d

    rq = RotorQuantMSE(d, bits, seed=42, device='cpu')
    iq_fast = IsoQuantMSE(d, bits, seed=42, mode='fast', device='cpu')
    pq = PlanarQuantMSE(d, bits, seed=42, device='cpu')

    return {
        'TQ+ PolarQuant': {'rotation_params': tqplus_rot_params, 'block_size': d},
        'RotorQuant': {'rotation_params': rq.rotors.numel(), 'block_size': 3},
        'IsoQuant-Fast': {'rotation_params': iq_fast.q_L.numel(), 'block_size': 4},
        'PlanarQuant': {'rotation_params': pq.rot2.numel(), 'block_size': 2},
    }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print()
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║  Cross-Validation: RotorQuant Family vs TurboQuant+ (TheTom)        ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == 'cuda' else ""))
    print(f"  TQ+: numpy (CPU)  |  Ours: PyTorch ({'GPU' if device == 'cuda' else 'CPU'})")
    print()

    dims = [64, 128, 256]
    bits_list = [2, 3, 4]

    # ── 1. Codebook comparison ──
    print("=" * 80)
    print("  1. LLOYD-MAX CODEBOOK COMPARISON")
    print("=" * 80)
    print(f"  {'d':>5} {'bits':>4} | {'Max centroid diff':>20} | {'TQ+ centroids':>30} | {'Ours':>30}")
    print("-" * 80)
    for d in dims:
        for bits in bits_list:
            diff, c_tq, c_ours = benchmark_codebook(d, bits)
            c_tq_str = ', '.join(f'{c:.4f}' for c in sorted(c_tq)[:4]) + '...'
            c_ours_str = ', '.join(f'{c:.4f}' for c in sorted(c_ours)[:4]) + '...'
            status = "MATCH" if diff < 1e-3 else f"DIFF={diff:.2e}"
            print(f"  {d:>5} {bits:>4} | {status:>20} | {c_tq_str:>30} | {c_ours_str:>30}")
    print()

    # ── 2. Reconstruction MSE ──
    print("=" * 80)
    print("  2. RECONSTRUCTION MSE (8192 unit vectors, lower is better)")
    print("=" * 80)
    print(f"  {'d':>5} {'bits':>4} | {'TQ+':>12} {'RotorQ':>12} {'IsoFast':>12} {'Planar':>12} | {'TQ+/RQ':>8} {'TQ+/Pln':>8}")
    print("-" * 80)
    for d in dims:
        for bits in bits_list:
            mses = benchmark_mse(d, bits, device=device)
            tq = mses['TQ+ PolarQuant']
            rq = mses['RotorQuant']
            iq = mses['IsoQuant-Fast']
            pq = mses['PlanarQuant']
            r_rq = tq / rq if rq > 0 else float('inf')
            r_pq = tq / pq if pq > 0 else float('inf')
            print(f"  {d:>5} {bits:>4} | {tq:>12.6f} {rq:>12.6f} {iq:>12.6f} {pq:>12.6f} | {r_rq:>7.3f}x {r_pq:>7.3f}x")
    print()

    # ── 3. Inner product preservation ──
    print("=" * 80)
    print("  3. INNER PRODUCT ESTIMATION (Stage 1 + Stage 2)")
    print("=" * 80)
    print(f"  {'d':>5} {'bits':>4} | {'TQ+ ip_mse':>12} {'RQ ip_mse':>12} {'IsoF ip_mse':>12} {'Pln ip_mse':>12} | {'TQ+ bias':>10} {'Pln bias':>10}")
    print("-" * 100)
    for d in dims:
        for bits in bits_list:
            ips = benchmark_inner_product(d, bits, device=device)
            tq = ips['TQ+ TurboQuant']
            rq = ips['RotorQuant']
            iq = ips['IsoQuant-Fast']
            pq = ips['PlanarQuant']
            print(f"  {d:>5} {bits:>4} | {tq['ip_mse']:>12.4f} {rq['ip_mse']:>12.4f} {iq['ip_mse']:>12.4f} {pq['ip_mse']:>12.4f} | {tq['ip_bias']:>+10.4f} {pq['ip_bias']:>+10.4f}")
    print()

    # ── 4. Speed ──
    if device == 'cuda':
        print("=" * 80)
        print("  4. SPEED (quantize + dequantize, 8192 vectors)")
        print("=" * 80)
        for d in [128]:
            for bits in [3]:
                speeds = benchmark_speed(d, bits, device=device)
                print(f"  d={d}, bits={bits}:")
                for name, t in speeds.items():
                    print(f"    {name:>25s}: {t:>10.0f} µs")
        print()

    # ── 5. Parameter count ──
    print("=" * 80)
    print("  5. PARAMETER COUNT (d=128, bits=3)")
    print("=" * 80)
    params = benchmark_params(128, 3)
    for name, info in params.items():
        print(f"  {name:>20s}: {info['rotation_params']:>6d} rotation params  (block_size={info['block_size']})")
    print()

    # ── Summary ──
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print("  If our methods match TQ+ on MSE and inner product quality,")
    print("  the rotation primitive (dense vs Clifford vs quaternion vs Givens)")
    print("  is a validated implementation with different speed/parameter tradeoffs.")
    print("  TQ+ uses numpy (CPU) with dense d×d rotation.")
    print("  Our methods use PyTorch (GPU) with block-wise rotation.")
    print()
