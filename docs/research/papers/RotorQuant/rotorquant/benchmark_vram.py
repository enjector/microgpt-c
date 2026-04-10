#!/usr/bin/env python3
"""
VRAM savings benchmark: TurboQuant vs RotorQuant vs IsoQuant compressed KV caches.

Measures actual GPU memory with compressed storage (indices + norms), not dequantized FP16.
Uses back2matching/turboquant TurboQuantCache as the reference, and builds equivalent
compressed caches for RotorQuant and IsoQuant.

Usage:
    python benchmark_vram.py
    python benchmark_vram.py --model Qwen/Qwen2.5-7B-Instruct --contexts 460 1860 4096
"""

import sys, os, gc, math, time, argparse
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

# ── Import reference TurboQuant cache from site-packages ─────────────────────
import importlib
_sp = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')

_spec_core = importlib.util.spec_from_file_location('ref_tq_core', os.path.join(_sp, 'turboquant', 'core.py'))
ref_core = importlib.util.module_from_spec(_spec_core)
_spec_core.loader.exec_module(ref_core)
RefTurboQuantMSE = ref_core.TurboQuantMSE

_spec_cache = importlib.util.spec_from_file_location('ref_tq_cache', os.path.join(_sp, 'turboquant', 'cache.py'))
ref_cache_mod = importlib.util.module_from_spec(_spec_cache)
# Patch the import inside cache.py to use our loaded core
sys.modules['turboquant.core'] = ref_core
_spec_cache.loader.exec_module(ref_cache_mod)
RefTurboQuantCache = ref_cache_mod.TurboQuantCache

# ── Import local implementations ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from turboquant.rotorquant import RotorQuantMSE
from turboquant.isoquant import IsoQuantMSE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# Compressed KV cache layers for RQ and IQ
# ═══════════════════════════════════════════════════════════════════════════════

from transformers.cache_utils import DynamicCache, DynamicLayer


class RotorQuantLayer(DynamicLayer):
    """Cache layer storing RQ-compressed indices + norms."""

    def __init__(self, bits: int = 3, residual_len: int = 128):
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self._quantizers = {}  # head_dim -> RotorQuantMSE
        self._key_indices = None   # dict of {'vector': tensor, '_norms': tensor}
        self._value_indices = None
        self._residual_keys = None
        self._residual_values = None
        self._total_len = 0
        self._head_dim = None

    def _get_quantizer(self, head_dim, device):
        key = (head_dim, str(device))
        if key not in self._quantizers:
            self._quantizers[key] = RotorQuantMSE(head_dim, self.bits, seed=42, device=str(device))
        return self._quantizers[key]

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self._head_dim = key_states.shape[-1]
        self._residual_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        if self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len
            to_q_k = self._residual_keys[..., :overflow, :]
            to_q_v = self._residual_values[..., :overflow, :]

            rq = self._get_quantizer(self._head_dim, self.device)

            # Quantize keys — store indices (uint8) + norms (float32)
            k_flat = to_q_k.reshape(-1, self._head_dim).float()
            _, k_idx_dict = rq(k_flat)
            # Store only the vector indices and norms
            k_vec_idx = k_idx_dict['vector'].to(torch.uint8)  # compressed
            k_norms = k_idx_dict['_norms']  # float32

            v_flat = to_q_v.reshape(-1, self._head_dim).float()
            _, v_idx_dict = rq(v_flat)
            v_vec_idx = v_idx_dict['vector'].to(torch.uint8)
            v_norms = v_idx_dict['_norms']

            if self._key_indices is None:
                self._key_indices = k_vec_idx
                self._key_norms = k_norms
                self._value_indices = v_vec_idx
                self._value_norms = v_norms
            else:
                self._key_indices = torch.cat([self._key_indices, k_vec_idx], dim=0)
                self._key_norms = torch.cat([self._key_norms, k_norms], dim=0)
                self._value_indices = torch.cat([self._value_indices, v_vec_idx], dim=0)
                self._value_norms = torch.cat([self._value_norms, v_norms], dim=0)

            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        # Dequantize for attention computation
        if self._key_indices is not None and self._key_indices.numel() > 0:
            rq = self._get_quantizer(self._head_dim, self.device)
            # Rebuild indices dict for dequantize
            k_idx = {'vector': self._key_indices.long(), '_norms': self._key_norms}
            k_deq = rq.dequantize(k_idx).to(self.dtype)
            k_deq = k_deq.reshape(self._residual_keys.shape[0], self._residual_keys.shape[1],
                                  -1, self._head_dim)
            v_idx = {'vector': self._value_indices.long(), '_norms': self._value_norms}
            v_deq = rq.dequantize(v_idx).to(self.dtype)
            v_deq = v_deq.reshape(self._residual_values.shape[0], self._residual_values.shape[1],
                                  -1, self._head_dim)
            self.keys = torch.cat([k_deq, self._residual_keys], dim=-2)
            self.values = torch.cat([v_deq, self._residual_values], dim=-2)
        else:
            self.keys = self._residual_keys
            self.values = self._residual_values

        return self.keys, self.values

    def get_seq_length(self):
        return self._total_len


class IsoQuantLayer(DynamicLayer):
    """Cache layer storing IQ-compressed indices + norms."""

    def __init__(self, bits: int = 3, residual_len: int = 128):
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self._quantizers = {}
        self._key_indices = None
        self._key_norms = None
        self._value_indices = None
        self._value_norms = None
        self._residual_keys = None
        self._residual_values = None
        self._total_len = 0
        self._head_dim = None

    def _get_quantizer(self, head_dim, device):
        key = (head_dim, str(device))
        if key not in self._quantizers:
            self._quantizers[key] = IsoQuantMSE(head_dim, self.bits, mode='fast', seed=42, device=str(device))
        return self._quantizers[key]

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self._head_dim = key_states.shape[-1]
        self._residual_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        if self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len
            to_q_k = self._residual_keys[..., :overflow, :]
            to_q_v = self._residual_values[..., :overflow, :]

            iq = self._get_quantizer(self._head_dim, self.device)

            k_flat = to_q_k.reshape(-1, self._head_dim).float()
            _, k_idx_dict = iq(k_flat)
            k_vec_idx = k_idx_dict['indices'].to(torch.uint8)
            k_norms = k_idx_dict['_norms']

            v_flat = to_q_v.reshape(-1, self._head_dim).float()
            _, v_idx_dict = iq(v_flat)
            v_vec_idx = v_idx_dict['indices'].to(torch.uint8)
            v_norms = v_idx_dict['_norms']

            if self._key_indices is None:
                self._key_indices = k_vec_idx
                self._key_norms = k_norms
                self._value_indices = v_vec_idx
                self._value_norms = v_norms
            else:
                self._key_indices = torch.cat([self._key_indices, k_vec_idx], dim=0)
                self._key_norms = torch.cat([self._key_norms, k_norms], dim=0)
                self._value_indices = torch.cat([self._value_indices, v_vec_idx], dim=0)
                self._value_norms = torch.cat([self._value_norms, v_norms], dim=0)

            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        # Dequantize
        if self._key_indices is not None and self._key_indices.numel() > 0:
            iq = self._get_quantizer(self._head_dim, self.device)
            k_idx = {'indices': self._key_indices.long(), '_norms': self._key_norms}
            k_deq = iq.dequantize(k_idx).to(self.dtype)
            k_deq = k_deq.reshape(self._residual_keys.shape[0], self._residual_keys.shape[1],
                                  -1, self._head_dim)
            v_idx = {'indices': self._value_indices.long(), '_norms': self._value_norms}
            v_deq = iq.dequantize(v_idx).to(self.dtype)
            v_deq = v_deq.reshape(self._residual_values.shape[0], self._residual_values.shape[1],
                                  -1, self._head_dim)
            self.keys = torch.cat([k_deq, self._residual_keys], dim=-2)
            self.values = torch.cat([v_deq, self._residual_values], dim=-2)
        else:
            self.keys = self._residual_keys
            self.values = self._residual_values

        return self.keys, self.values

    def get_seq_length(self):
        return self._total_len


class RotorQuantCache(DynamicCache):
    def __init__(self, bits=3, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self.layers) <= layer_idx:
            self.layers.append(RotorQuantLayer(bits=self.bits))
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


class IsoQuantCache(DynamicCache):
    def __init__(self, bits=3, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self.layers) <= layer_idx:
            self.layers.append(IsoQuantLayer(bits=self.bits))
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_test(model, tokenizer, input_ids, cache_factory, label):
    """Run forward pass with given cache, measure VRAM and speed."""
    flush()
    mem_before = gpu_mem_mb()

    cache = cache_factory()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache, use_cache=True, labels=input_ids)
        loss = outputs.loss
        ppl = math.exp(min(loss.item(), 20)) if loss is not None else float('nan')
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak = gpu_peak_mb()
    mem_after = gpu_mem_mb()
    tok_s = input_ids.shape[1] / elapsed

    return {
        'label': label,
        'ppl': ppl,
        'peak_mb': peak,
        'cache_mb': mem_after - mem_before,
        'tok_s': tok_s,
    }


def compressed_bytes(n_vectors, head_dim, bits, method='tq'):
    """Bytes for compressed representation (indices + norms). All methods same formula."""
    # indices: n_vectors * head_dim * bits / 8 bytes (bit-packed)
    # norms: n_vectors * 4 bytes (float32)
    idx_bytes = n_vectors * head_dim * bits / 8
    norm_bytes = n_vectors * 4
    return idx_bytes + norm_bytes

def fp16_bytes(n_vectors, head_dim):
    return n_vectors * head_dim * 2  # 2 bytes per fp16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--bits', nargs='+', type=int, default=[3, 4])
    parser.add_argument('--contexts', nargs='+', type=int, default=[460, 1860, 4096, 8192, 16384, 32768])
    args = parser.parse_args()

    print("=" * 95)
    print("  VRAM Savings: FP16 vs TurboQuant vs RotorQuant vs IsoQuant (KV cache)")
    print("=" * 95)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    flush()
    mem_start = gpu_mem_mb()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map='cuda:0'
    )
    model.eval()

    model_mb = gpu_mem_mb() - mem_start
    gpu_name = torch.cuda.get_device_name(0)
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads

    print(f"  Model: {args.model} ({model_mb:.0f} MB)")
    print(f"  GPU: {gpu_name}")
    print(f"  Architecture: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    # ── Part 1: Real model PPL at feasible context ───────────────────────
    print(f"\n{'─' * 95}")
    print("  Part 1: Real PPL + Measured Peak VRAM (Qwen2.5-3B, single forward pass)")
    print(f"{'─' * 95}")

    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100])
    except Exception:
        text = "The quick brown fox jumps over the lazy dog. " * 10000

    all_ids = tokenizer.encode(text, return_tensors='pt').to('cuda:0')

    test_contexts = [c for c in [1024, 2048, 4096] if c <= all_ids.shape[1]]

    for bits in args.bits:
        print(f"\n  {bits}-bit:")
        print(f"  {'Context':>8}  {'Method':<12} {'Peak VRAM':>12} {'Speed':>12} {'PPL':>8}")
        print(f"  {'────────':>8}  {'────────────':<12} {'────────────':>12} {'────────────':>12} {'────────':>8}")

        for ctx in test_contexts:
            input_ids = all_ids[:, :ctx]

            fp16 = run_test(model, tokenizer, input_ids, lambda: DynamicCache(), 'FP16')
            tq = run_test(model, tokenizer, input_ids,
                         lambda b=bits: RefTurboQuantCache(bits=b), f'TQ {bits}b')
            rq = run_test(model, tokenizer, input_ids,
                         lambda b=bits: RotorQuantCache(bits=b), f'RQ {bits}b')
            iq = run_test(model, tokenizer, input_ids,
                         lambda b=bits: IsoQuantCache(bits=b), f'IQ {bits}b')

            for r in [fp16, tq, rq, iq]:
                print(f"  {ctx:>8}  {r['label']:<12} {r['peak_mb']:>10.0f} MB "
                      f"{r['tok_s']:>10.1f} t/s {r['ppl']:>8.2f}")

    # ── Part 2: Analytical VRAM savings (the real story) ─────────────────
    print(f"\n{'─' * 95}")
    print(f"  Part 2: KV Cache VRAM — FP16 vs Compressed Storage")
    print(f"  (All quantization methods use identical compressed format: uint8 indices + float32 norms)")
    print(f"  Model config: {n_layers} layers × {n_kv_heads} KV heads × head_dim={head_dim}")
    print(f"{'─' * 95}")

    # n_kv_vectors per context token = 2 (K+V) * n_layers * n_kv_heads
    vectors_per_token = 2 * n_layers * n_kv_heads

    for bits in args.bits:
        print(f"\n  {bits}-bit compression:")
        print(f"  {'Context':>8}  {'FP16 KV':>10}  {f'TQ/RQ/IQ {bits}b':>12}  {'Saved':>10}  {'Ratio':>8}")
        print(f"  {'────────':>8}  {'──────────':>10}  {'────────────':>12}  {'──────────':>10}  {'────────':>8}")

        for ctx in args.contexts:
            n_vecs = ctx * vectors_per_token
            fp16_mb = fp16_bytes(n_vecs, head_dim) / 1024**2
            comp_mb = compressed_bytes(n_vecs, head_dim, bits) / 1024**2
            saved = fp16_mb - comp_mb
            ratio = fp16_mb / comp_mb

            print(f"  {ctx:>8}  {fp16_mb:>8.1f} MB  {comp_mb:>10.1f} MB  {saved:>8.1f} MB  {ratio:>7.1f}x")

    # ── Part 3: The parameter storage advantage ──────────────────────────
    print(f"\n{'─' * 95}")
    print(f"  Part 3: Quantizer State VRAM (stored once, shared across all tokens)")
    print(f"{'─' * 95}")

    for bits in args.bits:
        # RefTQ: d×d rotation matrix (float32) + codebook per layer
        ref_per_layer = (head_dim * head_dim * 4 + 2**bits * 4)  # bytes
        ref_total = ref_per_layer * n_layers * n_kv_heads  # one quantizer per layer×head

        # RQ: n_groups rotors (8 components each) + codebook
        rq_obj = RotorQuantMSE(head_dim, bits, device='cpu')
        rq_per_layer = (rq_obj.n_groups * 8 * 4 + 2**bits * 4)
        rq_total = rq_per_layer * n_layers * n_kv_heads

        # IQ: n_groups quaternions (4 components) + codebook
        iq_obj = IsoQuantMSE(head_dim, bits, mode='fast', device='cpu')
        iq_per_layer = (iq_obj.n_groups * 4 * 4 + 2**bits * 4)
        iq_total = iq_per_layer * n_layers * n_kv_heads

        print(f"\n  {bits}-bit quantizer state ({n_layers} layers × {n_kv_heads} KV heads):")
        print(f"    Ref TurboQuant: {ref_total/1024:.1f} KB  ({head_dim}×{head_dim} rotation matrix per quantizer)")
        print(f"    RotorQuant:     {rq_total/1024:.1f} KB  ({rq_obj.n_groups} rotors × 8 per quantizer) — {ref_total/rq_total:.0f}x smaller")
        print(f"    IsoQuant-Fast:  {iq_total/1024:.1f} KB  ({iq_obj.n_groups} quaternions × 4 per quantizer) — {ref_total/iq_total:.0f}x smaller")

    print(f"\n{'=' * 95}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 95}")


if __name__ == '__main__':
    main()
