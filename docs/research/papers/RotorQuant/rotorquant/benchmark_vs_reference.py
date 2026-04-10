#!/usr/bin/env python3
"""
Head-to-head benchmark: back2matching/turboquant (reference) vs RotorQuant/IsoQuant.

Compares:
  1. Synthetic MSE distortion (unit vectors, d=128)
  2. Inner product preservation (QJL two-stage)
  3. NIAH retrieval
  4. Real model PPL + VRAM (Qwen2.5-3B-Instruct)
  5. Speed (quantize+dequantize latency)
  6. VRAM savings at different context lengths

Reference: pip install turboquant (back2matching, v0.2.0)
"""

import sys, os, time, math, gc, argparse
import torch
import numpy as np

# ── Import reference turboquant from site-packages (avoid local shadow) ──────
import importlib
_sp = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
_spec_core = importlib.util.spec_from_file_location('ref_tq_core', os.path.join(_sp, 'turboquant', 'core.py'))
ref_core = importlib.util.module_from_spec(_spec_core)
_spec_core.loader.exec_module(ref_core)
RefTurboQuantMSE = ref_core.TurboQuantMSE
RefTurboQuantIP = ref_core.TurboQuantIP

# ── Import local implementations ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from turboquant.rotorquant import RotorQuantMSE, RotorQuantProd
from turboquant.isoquant import IsoQuantMSE, IsoQuantProd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'


def gpu_mem_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def gpu_peak_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: MSE Distortion
# ═══════════════════════════════════════════════════════════════════════════════

def test_mse(d=128, n=2000, bits_list=[2, 3, 4]):
    print("\n" + "=" * 72)
    print("TEST 1: MSE Distortion — Reference TQ vs RotorQuant vs IsoQuant")
    print("=" * 72)
    print(f"  d={d}, n_vectors={n}, device={DEVICE}\n")

    # Generate random unit vectors
    torch.manual_seed(42)
    x = torch.randn(n, d, device=DEVICE)
    x = x / x.norm(dim=-1, keepdim=True)

    # Theoretical bound: (d/(d-1)) * Beta(d/2, 1/2) / (4^b * d^(1/2))
    # Simplified: sqrt(3)*pi/2 * 1/4^b (from paper)
    def theory(b):
        return math.sqrt(3) * math.pi / 2 * (1 / 4**b)

    print(f"  {'bits':>4}  {'Ref TQ MSE':>12}  {'RQ MSE':>12}  {'IQ-Fast MSE':>12}  {'theory':>12}  {'winner':>8}")
    print(f"  {'────':>4}  {'────────────':>12}  {'────────────':>12}  {'────────────':>12}  {'────────────':>12}  {'────────':>8}")

    for bits in bits_list:
        # Reference TurboQuant
        ref_tq = RefTurboQuantMSE(d, bits=bits, device=DEVICE, seed=42)
        idx, norms = ref_tq.quantize(x)
        x_ref = ref_tq.dequantize(idx, norms)
        mse_ref = ((x - x_ref)**2).sum(dim=-1).mean().item()

        # RotorQuant
        rq = RotorQuantMSE(d, bits, seed=42, device=DEVICE)
        x_rq, _ = rq(x)
        mse_rq = ((x - x_rq)**2).sum(dim=-1).mean().item()

        # IsoQuant-Fast
        iq = IsoQuantMSE(d, bits, mode='fast', seed=42, device=DEVICE)
        x_iq, _ = iq(x)
        mse_iq = ((x - x_iq)**2).sum(dim=-1).mean().item()

        th = theory(bits)
        best = min(mse_ref, mse_rq, mse_iq)
        winner = 'RefTQ' if best == mse_ref else ('RQ' if best == mse_rq else 'IQ')

        print(f"  {bits:>4}  {mse_ref:>12.6f}  {mse_rq:>12.6f}  {mse_iq:>12.6f}  {th:>12.6f}  {winner:>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Inner Product Preservation
# ═══════════════════════════════════════════════════════════════════════════════

def test_inner_product(d=128, n=2000, n_pairs=3000, bits_list=[2, 3, 4]):
    print("\n" + "=" * 72)
    print("TEST 2: Inner Product Unbiasedness (two-stage with QJL)")
    print("=" * 72)
    print(f"  d={d}, n_pairs={n_pairs}\n")

    torch.manual_seed(42)
    x = torch.randn(n_pairs, d, device=DEVICE)
    y = torch.randn(n_pairs, d, device=DEVICE)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    true_ip = (x * y).sum(dim=-1)

    print(f"  {'bits':>4}  {'method':>8}  {'bias':>12}  {'RMSE':>12}  {'corr':>8}")
    print(f"  {'────':>4}  {'────────':>8}  {'────────────':>12}  {'────────────':>12}  {'────────':>8}")

    for bits in bits_list:
        results = {}

        # Reference TQ IP (two-stage)
        ref_ip = RefTurboQuantIP(d, bits=bits, device=DEVICE, seed=42)
        cx = ref_ip.quantize(x)
        cy = ref_ip.quantize(y)
        x_hat = ref_ip.dequantize(*cx)
        y_hat = ref_ip.dequantize(*cy)
        approx = (x_hat * y_hat).sum(dim=-1)
        results['RefTQ'] = approx

        # RotorQuant Prod — uses dict-based API with inner_product method
        rqp = RotorQuantProd(d, bits, seed=42, device=DEVICE)
        cx_rq = rqp(x)  # returns dict
        x_rq_hat = rqp.dequantize(cx_rq)
        cy_rq = rqp(y)
        y_rq_hat = rqp.dequantize(cy_rq)
        approx_rq = (x_rq_hat * y_rq_hat).sum(dim=-1)
        results['RQ'] = approx_rq

        # IsoQuant Prod
        iqp = IsoQuantProd(d, bits, mode='fast', seed=42, device=DEVICE)
        cx_iq = iqp(x)
        x_iq_hat = iqp.dequantize(cx_iq)
        cy_iq = iqp(y)
        y_iq_hat = iqp.dequantize(cy_iq)
        approx_iq = (x_iq_hat * y_iq_hat).sum(dim=-1)
        results['IQ'] = approx_iq

        for name, approx in results.items():
            diff = approx - true_ip
            bias = diff.mean().item()
            rmse = diff.pow(2).mean().sqrt().item()
            corr = torch.corrcoef(torch.stack([true_ip, approx]))[0, 1].item()
            print(f"  {bits:>4}  {name:>8}  {bias:>+12.6f}  {rmse:>12.6f}  {corr:>8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: NIAH Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def test_niah(d=128, bits_list=[2, 3, 4], seq_lens=[512, 2048, 8192]):
    print("\n" + "=" * 72)
    print("TEST 3: Needle-in-Haystack Retrieval")
    print("=" * 72)

    print(f"\n  {'bits':>4}  {'seq':>6}  {'RefTQ':>8}  {'RQ':>8}  {'IQ':>8}")
    print(f"  {'────':>4}  {'──────':>6}  {'────────':>8}  {'────────':>8}  {'────────':>8}")

    for bits in bits_list:
        for seq_len in seq_lens:
            torch.manual_seed(42)
            keys = torch.randn(seq_len, d, device=DEVICE)
            keys = keys / keys.norm(dim=-1, keepdim=True)
            needle_idx = seq_len // 3
            needle = keys[needle_idx].clone()
            query = needle + 0.01 * torch.randn(d, device=DEVICE)
            query = query / query.norm()

            results = {}

            # Reference TQ
            ref_tq = RefTurboQuantMSE(d, bits=bits, device=DEVICE, seed=42)
            idx, norms = ref_tq.quantize(keys)
            k_hat = ref_tq.dequantize(idx, norms)
            scores = k_hat @ query
            found = scores.argmax().item()
            results['RefTQ'] = 'EXACT' if found == needle_idx else f'MISS({abs(found-needle_idx)})'

            # RotorQuant
            rq = RotorQuantMSE(d, bits, seed=42, device=DEVICE)
            k_rq, _ = rq(keys)
            scores_rq = k_rq @ query
            found_rq = scores_rq.argmax().item()
            results['RQ'] = 'EXACT' if found_rq == needle_idx else f'MISS({abs(found_rq-needle_idx)})'

            # IsoQuant
            iq = IsoQuantMSE(d, bits, mode='fast', seed=42, device=DEVICE)
            k_iq, _ = iq(keys)
            scores_iq = k_iq @ query
            found_iq = scores_iq.argmax().item()
            results['IQ'] = 'EXACT' if found_iq == needle_idx else f'MISS({abs(found_iq-needle_idx)})'

            print(f"  {bits:>4}  {seq_len:>6}  {results['RefTQ']:>8}  {results['RQ']:>8}  {results['IQ']:>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Speed Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def test_speed(d=128, bits=3, n_list=[1000, 5000, 10000]):
    print("\n" + "=" * 72)
    print("TEST 4: Speed Benchmark (quantize + dequantize roundtrip)")
    print("=" * 72)
    print(f"  GPU: {GPU_NAME}")
    print(f"  d={d}, bits={bits}\n")

    for n in n_list:
        torch.manual_seed(42)
        x = torch.randn(n, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        # Warmup
        for _ in range(3):
            ref_tq = RefTurboQuantMSE(d, bits=bits, device=DEVICE, seed=42)
            idx, norms = ref_tq.quantize(x)
            ref_tq.dequantize(idx, norms)
            torch.cuda.synchronize()

        # Reference TQ
        t0 = time.perf_counter()
        for _ in range(10):
            idx, norms = ref_tq.quantize(x)
            _ = ref_tq.dequantize(idx, norms)
            torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / 10 * 1000

        # RotorQuant
        rq = RotorQuantMSE(d, bits, seed=42, device=DEVICE)
        for _ in range(3):
            rq(x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            rq(x); torch.cuda.synchronize()
        t_rq = (time.perf_counter() - t0) / 10 * 1000

        # IsoQuant-Fast
        iq = IsoQuantMSE(d, bits, mode='fast', seed=42, device=DEVICE)
        for _ in range(3):
            iq(x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            iq(x); torch.cuda.synchronize()
        t_iq = (time.perf_counter() - t0) / 10 * 1000

        print(f"  n={n:>5}: RefTQ={t_ref:>8.2f}ms  RQ={t_rq:>8.2f}ms  IQ={t_iq:>8.2f}ms"
              f"  (RefTQ/RQ={t_ref/max(t_rq,0.001):.1f}x  RefTQ/IQ={t_ref/max(t_iq,0.001):.1f}x)")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Parameter Efficiency
# ═══════════════════════════════════════════════════════════════════════════════

def test_params(d=128, bits=3):
    print("\n" + "=" * 72)
    print("TEST 5: Parameter / Buffer Efficiency")
    print("=" * 72)

    # Reference TQ
    ref_tq = RefTurboQuantMSE(d, bits=bits, device='cpu', seed=42)
    ref_params = d * d + 2**bits  # rotation matrix + codebook
    print(f"\n  Reference TurboQuant (back2matching):")
    print(f"    Rotation matrix: {d}x{d} = {d*d:,}")
    print(f"    Codebook: {2**bits} centroids")
    print(f"    Total: {ref_params:,}")

    # RotorQuant
    rq = RotorQuantMSE(d, bits, seed=42, device='cpu')
    n_groups = rq.n_groups
    rq_params = n_groups * 8 + 2**bits  # rotors + codebook
    print(f"\n  RotorQuant (Clifford Cl(3,0)):")
    print(f"    Rotors: {n_groups} groups x 8 components = {n_groups*8}")
    print(f"    Codebook: {2**bits} centroids")
    print(f"    Total: {rq_params}")
    print(f"    Ratio: {ref_params/rq_params:.1f}x smaller")

    # IsoQuant
    iq = IsoQuantMSE(d, bits, mode='fast', seed=42, device='cpu')
    n_groups_iq = iq.n_groups
    iq_params = n_groups_iq * 4 + 2**bits  # quaternions + codebook
    print(f"\n  IsoQuant-Fast (quaternion 4D):")
    print(f"    Quaternions: {n_groups_iq} groups x 4 components = {n_groups_iq*4}")
    print(f"    Codebook: {2**bits} centroids")
    print(f"    Total: {iq_params}")
    print(f"    Ratio: {ref_params/iq_params:.1f}x smaller")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Real Model PPL + VRAM
# ═══════════════════════════════════════════════════════════════════════════════

def test_real_model_ppl(model_name='Qwen/Qwen2.5-3B-Instruct', bits_list=[3, 4],
                        n_tokens=512, prefill_len=256):
    print("\n" + "=" * 72)
    print("TEST 6: Real Model PPL + VRAM (post-prefill K-cache quantization)")
    print("=" * 72)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    print(f"  Model: {model_name}")
    print(f"  Tokens: {n_tokens} (prefill={prefill_len})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()
    mem_before_model = gpu_mem_mb()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map='auto'
    )
    model.eval()

    mem_after_model = gpu_mem_mb()
    print(f"  Model VRAM: {mem_after_model - mem_before_model:.0f} MB")

    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    print(f"  Architecture: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    # Load wikitext
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100])
    except Exception:
        text = "The quick brown fox " * 5000

    input_ids = tokenizer.encode(text, return_tensors='pt')[:, :n_tokens].to(model.device)

    # ── Helper: single-pass PPL with monkey-patched cache ────────────────
    def run_ppl(quantize_fn=None, label='FP16'):
        """
        Single forward pass with monkey-patched DynamicCache.update.
        Quantizes K vectors as they enter the cache (simulates online compression).
        """
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        _orig = DynamicCache.update

        if quantize_fn is not None:
            def _patched(self, ks, vs, li, ck=None):
                # Quantize keys before storing
                kq = quantize_fn(ks, li)
                return _orig(self, kq, vs, li, ck)
            DynamicCache.update = _patched

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            ppl = math.exp(min(outputs.loss.item(), 20))  # cap at exp(20) to avoid overflow

        if quantize_fn is not None:
            DynamicCache.update = _orig

        peak = gpu_peak_mb()
        vram_kv = peak - mem_after_model
        return ppl, peak, vram_kv

    # ── Build quantize functions ─────────────────────────────────────────
    def make_ref_compress(bits):
        quantizers = {}
        def compress(ks, li):
            D = ks.shape[-1]
            if li not in quantizers:
                quantizers[li] = RefTurboQuantMSE(D, bits=bits, device=str(ks.device), seed=li*1000)
            tq = quantizers[li]
            flat = ks.reshape(-1, D).float()
            idx, norms = tq.quantize(flat)
            x_hat = tq.dequantize(idx, norms)
            return x_hat.to(ks.dtype).reshape(ks.shape)
        return compress

    def make_rq_compress(bits):
        quantizers = {}
        def compress(ks, li):
            D = ks.shape[-1]
            if li not in quantizers:
                quantizers[li] = RotorQuantMSE(D, bits, seed=li*1000, device=str(ks.device))
            rq = quantizers[li]
            flat = ks.reshape(-1, D).float()
            x_hat, _ = rq(flat)
            return x_hat.to(ks.dtype).reshape(ks.shape)
        return compress

    def make_iq_compress(bits):
        quantizers = {}
        def compress(ks, li):
            D = ks.shape[-1]
            if li not in quantizers:
                quantizers[li] = IsoQuantMSE(D, bits, mode='fast', seed=li*1000, device=str(ks.device))
            iq = quantizers[li]
            flat = ks.reshape(-1, D).float()
            x_hat, _ = iq(flat)
            return x_hat.to(ks.dtype).reshape(ks.shape)
        return compress

    # ── Run tests ────────────────────────────────────────────────────────
    ppl_fp16, peak_fp16, kv_fp16 = run_ppl(None, 'FP16')
    print(f"\n  {'Method':<20} {'PPL':>8} {'vs FP16':>10} {'Peak VRAM':>12} {'KV est.':>10}")
    print(f"  {'──────────────────':<20} {'────────':>8} {'──────────':>10} {'────────────':>12} {'──────────':>10}")
    print(f"  {'FP16 baseline':<20} {ppl_fp16:>8.2f} {'--':>10} {peak_fp16:>10.0f} MB {kv_fp16:>8.0f} MB")

    for bits in bits_list:
        ppl_ref, peak_ref, kv_ref = run_ppl(make_ref_compress(bits), f'RefTQ {bits}b')
        ppl_rq, peak_rq, kv_rq = run_ppl(make_rq_compress(bits), f'RQ {bits}b')
        ppl_iq, peak_iq, kv_iq = run_ppl(make_iq_compress(bits), f'IQ {bits}b')

        print(f"  {'RefTQ '+str(bits)+'b':<20} {ppl_ref:>8.2f} {'+'+f'{ppl_ref-ppl_fp16:.2f}':>10} {peak_ref:>10.0f} MB {kv_ref:>8.0f} MB")
        print(f"  {'RotorQ '+str(bits)+'b':<20} {ppl_rq:>8.2f} {'+'+f'{ppl_rq-ppl_fp16:.2f}':>10} {peak_rq:>10.0f} MB {kv_rq:>8.0f} MB")
        print(f"  {'IsoQ-F '+str(bits)+'b':<20} {ppl_iq:>8.2f} {'+'+f'{ppl_iq-ppl_fp16:.2f}':>10} {peak_iq:>10.0f} MB {kv_iq:>8.0f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: VRAM Savings at Scale (projected, no model needed)
# ═══════════════════════════════════════════════════════════════════════════════

def test_vram_projection():
    print("\n" + "=" * 72)
    print("TEST 7: Projected VRAM Savings (analytical)")
    print("=" * 72)
    print("  Based on Qwen2.5-7B: 28 layers, 4 KV heads, head_dim=128\n")

    n_layers, n_kv_heads, head_dim = 28, 4, 128

    def kv_bytes_fp16(seq_len):
        # 2 (K+V) * n_layers * n_kv_heads * seq_len * head_dim * 2 bytes
        return 2 * n_layers * n_kv_heads * seq_len * head_dim * 2

    def kv_bytes_quant(seq_len, bits):
        # Quantized: bits per coordinate + 32-bit norm per vector
        per_vector_bits = head_dim * bits + 32  # indices + norm
        per_vector_bytes = per_vector_bits / 8
        return 2 * n_layers * n_kv_heads * seq_len * per_vector_bytes

    print(f"  {'Context':>8}  {'FP16 KV':>10}  {'4-bit KV':>10}  {'3-bit KV':>10}  {'4b saved':>10}  {'3b saved':>10}  {'4b ratio':>8}  {'3b ratio':>8}")
    print(f"  {'────────':>8}  {'──────────':>10}  {'──────────':>10}  {'──────────':>10}  {'──────────':>10}  {'──────────':>10}  {'────────':>8}  {'────────':>8}")

    for ctx in [460, 1860, 4096, 8192, 16384, 32768, 65536, 131072]:
        fp16 = kv_bytes_fp16(ctx) / 1024**2
        q4 = kv_bytes_quant(ctx, 4) / 1024**2
        q3 = kv_bytes_quant(ctx, 3) / 1024**2
        print(f"  {ctx:>8}  {fp16:>8.1f}MB  {q4:>8.1f}MB  {q3:>8.1f}MB  {fp16-q4:>8.1f}MB  {fp16-q3:>8.1f}MB  {fp16/q4:>7.1f}x  {fp16/q3:>7.1f}x")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RotorQuant vs Reference TurboQuant')
    parser.add_argument('--bits', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--skip-model', action='store_true', help='Skip real model PPL test')
    parser.add_argument('--skip-synthetic', action='store_true', help='Skip synthetic tests')
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  RotorQuant vs Reference TurboQuant (back2matching v0.2.0)       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"  GPU: {GPU_NAME}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Bits: {args.bits}")

    if not args.skip_synthetic:
        test_mse(bits_list=args.bits)
        test_inner_product(bits_list=args.bits)
        test_niah(bits_list=args.bits)
        test_speed()
        test_params()

    test_vram_projection()

    if not args.skip_model:
        test_real_model_ppl(model_name=args.model, bits_list=[b for b in args.bits if b >= 3])

    print("\n" + "=" * 72)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 72)
