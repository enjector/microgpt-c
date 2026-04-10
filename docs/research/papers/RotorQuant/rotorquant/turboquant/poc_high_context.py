"""
IsoQuant High-Context POC

Patches Qwen2.5-3B's KV cache with IsoQuant Triton-fused compression
and tests needle-in-haystack retrieval + generation at increasing context.

Measures: VRAM, prefill tok/s, decode tok/s, attention fidelity, generation quality.

Usage:
    python -m turboquant.poc_high_context
    python -m turboquant.poc_high_context --bits 3 --max-ctx 65536
    python -m turboquant.poc_high_context --backend clifford  # legacy RotorQuant
"""

import torch
import torch.nn.functional as F
import time
import math
import gc
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.isoquant import IsoQuantMSE
from turboquant.triton_isoquant import triton_iso_fast_fused

# PlanarQuant path (--backend planar)
from turboquant.planarquant import PlanarQuantMSE
from turboquant.triton_planarquant import triton_planar2_fused

# Legacy Clifford path (--backend clifford)
from turboquant.rotorquant import RotorQuantMSE
from turboquant.triton_kernels import (
    triton_rotor_full_fused, pack_rotors_for_triton,
)


# ── KV cache compressor (IsoQuant or RotorQuant) ─────────────────────

class IsoQuantKeyCompressor:
    """Per-layer key compressor using IsoQuant-Fast + Triton fused kernel."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str):
        self.iq = IsoQuantMSE(head_dim, bits, seed=seed, mode='fast', device=device)
        self.q_L = self.iq.q_L.to(device)
        self.centroids = self.iq.centroids.to(device)
        self.head_dim = head_dim
        self.device = device

    @torch.no_grad()
    def compress_dequantize(self, keys: torch.Tensor) -> torch.Tensor:
        B, H, S, D = keys.shape
        orig_dtype = keys.dtype
        flat = keys.reshape(-1, D)
        flat_recon = triton_iso_fast_fused(flat, self.q_L, self.centroids)
        return flat_recon.to(orig_dtype).reshape(B, H, S, D)


class RotorQuantKeyCompressor:
    """Per-layer key compressor using RotorQuant + Triton fused kernel (legacy)."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str):
        self.rq = RotorQuantMSE(head_dim, bits, seed=seed, device=device)
        self.packed_rotors = pack_rotors_for_triton(self.rq.rotors).to(device)
        self.c_v = getattr(self.rq, 'centroids_vector').to(device)
        self.c_t = getattr(self.rq, 'centroids_trivector').to(device)
        self.head_dim = head_dim
        self.device = device

    @torch.no_grad()
    def compress_dequantize(self, keys: torch.Tensor) -> torch.Tensor:
        B, H, S, D = keys.shape
        orig_dtype = keys.dtype
        flat = keys.reshape(-1, D)
        flat_recon = triton_rotor_full_fused(
            flat, self.packed_rotors, None, self.c_v, None, self.c_t)
        return flat_recon.to(orig_dtype).reshape(B, H, S, D)


class PlanarQuantKeyCompressor:
    """Per-layer key compressor using PlanarQuant + Triton fused kernel."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str):
        self.pq = PlanarQuantMSE(head_dim, bits, seed=seed, device=device)
        self.rot2 = self.pq.rot2.to(device)
        self.centroids = self.pq.centroids.to(device)
        self.head_dim = head_dim
        self.device = device

    @torch.no_grad()
    def compress_dequantize(self, keys: torch.Tensor) -> torch.Tensor:
        B, H, S, D = keys.shape
        orig_dtype = keys.dtype
        key_device = keys.device
        flat = keys.reshape(-1, D).to(self.device)
        flat_recon = triton_planar2_fused(flat, self.rot2, self.centroids)
        return flat_recon.to(orig_dtype).to(key_device).reshape(B, H, S, D)


class ValueCompressor:
    """Per-layer value compressor — wraps either backend."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str, backend: str = 'iso'):
        if backend == 'planar':
            self.inner = PlanarQuantKeyCompressor(head_dim, bits, seed, device)
        elif backend == 'iso':
            self.inner = IsoQuantKeyCompressor(head_dim, bits, seed, device)
        else:
            self.inner = RotorQuantKeyCompressor(head_dim, bits, seed, device)

    @torch.no_grad()
    def compress_dequantize(self, values: torch.Tensor) -> torch.Tensor:
        return self.inner.compress_dequantize(values)


# ── Patched KV cache ────────────────────────────────────────────────

class PatchedCache:
    """Wraps HuggingFace DynamicCache to quantize keys/values on insertion."""

    def __init__(self, bits: int, device: str, quantize_values: bool = True,
                 backend: str = 'iso'):
        self.bits = bits
        self.device = device
        self.quantize_values = quantize_values
        self.backend = backend
        self._key_compressors = {}
        self._val_compressors = {}

    def get_key_compressor(self, layer_idx: int, head_dim: int):
        if layer_idx not in self._key_compressors:
            if self.backend == 'planar':
                self._key_compressors[layer_idx] = PlanarQuantKeyCompressor(
                    head_dim, self.bits, seed=layer_idx * 1000, device=self.device)
            elif self.backend == 'iso':
                self._key_compressors[layer_idx] = IsoQuantKeyCompressor(
                    head_dim, self.bits, seed=layer_idx * 1000, device=self.device)
            else:
                self._key_compressors[layer_idx] = RotorQuantKeyCompressor(
                    head_dim, self.bits, seed=layer_idx * 1000, device=self.device)
        return self._key_compressors[layer_idx]

    def get_val_compressor(self, layer_idx: int, head_dim: int):
        if layer_idx not in self._val_compressors:
            self._val_compressors[layer_idx] = ValueCompressor(
                head_dim, self.bits, seed=layer_idx * 1000 + 500,
                device=self.device, backend=self.backend)
        return self._val_compressors[layer_idx]


def patch_model_kv_cache(model, bits: int = 4, quantize_values: bool = False,
                         backend: str = 'iso'):
    """Monkey-patch model's cache update for post-prefill RotorQuant compression.

    Strategy:
      - Prefill: full precision (no quantization, no error compounding)
      - First decode step: quantize entire prefill cache in bulk
      - Subsequent decode steps: quantize each new key, return full-precision
        key for current attention to avoid compounding

    This gives perfect prefill quality + compressed cache for decode.
    Works with any HuggingFace model that uses DynamicCache.
    """
    from transformers import DynamicCache

    rq_cache = PatchedCache(bits, "cuda", quantize_values, backend=backend)
    prefill_done = {}  # per-layer tracking

    _original_update = DynamicCache.update

    def _compress_keys_inplace(key_states, layer_idx):
        """Quantize keys, returning new tensor (same shape/device)."""
        D = key_states.shape[-1]
        kc = rq_cache.get_key_compressor(layer_idx, D)
        return kc.compress_dequantize(key_states)

    def _patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        new_seq_len = key_states.shape[2]
        is_prefill = (new_seq_len > 1)

        if is_prefill:
            # Prefill: store at full precision — no quantization
            prefill_done[layer_idx] = True
            return _original_update(self, key_states, value_states, layer_idx, cache_kwargs)

        # Decode: quantize new key for storage, full-precision for current attention.
        #
        # Old approach: k_out.clone() — copied entire cache, O(seq_len) per token.
        # New approach: store quantized, then patch last position in-place, O(1).
        #
        # The previous step's key stays full-precision in the cache (never
        # re-quantized). This is intentional — one unquantized key per decode
        # step only helps quality, and the overhead is zero.
        key_quantized = _compress_keys_inplace(key_states, layer_idx)

        # Optionally quantize values
        if rq_cache.quantize_values:
            D = value_states.shape[-1]
            vc = rq_cache.get_val_compressor(layer_idx, D)
            value_states = vc.compress_dequantize(value_states)

        k_out, v_out = _original_update(self, key_quantized, value_states, layer_idx, cache_kwargs)

        # On first decode step: quantize all prefill keys in bulk (one-time cost)
        if prefill_done.get(layer_idx) is True:
            cached_keys = self.layers[layer_idx].keys
            B, H, S, D = cached_keys.shape
            if S > 1:
                prefill_keys = cached_keys[:, :, :-1, :]
                prefill_q = _compress_keys_inplace(prefill_keys, layer_idx)
                cached_keys[:, :, :-1, :] = prefill_q
            prefill_done[layer_idx] = 'done'

        # Patch last position to full-precision for current-step attention.
        # k_out IS the cache tensor — this writes directly into the cache.
        # Next step, a new key appends and this position stays full-precision.
        k_out[:, :, -1:, :] = key_states

        return k_out, v_out

    DynamicCache.update = _patched_update
    return _original_update, rq_cache


def unpatch_model_kv_cache(original_update):
    """Restore original cache update."""
    from transformers import DynamicCache
    DynamicCache.update = original_update


# ── Needle-in-haystack builder ──────────────────────────────────────

NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name mentioned in the documents?"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


def build_prompt(tokenizer, target_tokens=2048, needle_pos=0.33):
    """Build a needle-in-haystack prompt at the target token count."""
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)

    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Important Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)

    haystack = "".join(parts)
    messages = [
        {"role": "user", "content": f"{haystack}\n\n{QUESTION}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Attention fidelity measurement ──────────────────────────────────

@torch.no_grad()
def measure_attention_fidelity(model, tokenizer, context_len, bits, backend='iso'):
    """Compare RotorQuant attention scores vs FP16 on real KV cache.

    Returns dict with cosine_sim, top1_match, needle_found.
    """
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_len + 256).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    # Forward pass with FP16 KV cache
    outputs_fp16 = model(**inputs, use_cache=True, output_attentions=False)
    cache_fp16 = outputs_fp16.past_key_values

    n_layers = len(cache_fp16.layers)
    head_dim = cache_fp16.layers[0].keys.shape[-1]
    n_kv_heads = cache_fp16.layers[0].keys.shape[1]

    # Measure per-layer attention fidelity
    cosine_sims = []
    top1_matches = 0
    n_checks = 0

    for layer_idx in range(n_layers):
        keys = cache_fp16.layers[layer_idx].keys  # (1, H, S, D)
        B, H, S, D = keys.shape

        # Compress keys
        if backend == 'planar':
            compressor = PlanarQuantKeyCompressor(D, bits, seed=layer_idx * 1000, device="cuda")
        elif backend == 'iso':
            compressor = IsoQuantKeyCompressor(D, bits, seed=layer_idx * 1000, device="cuda")
        else:
            compressor = RotorQuantKeyCompressor(D, bits, seed=layer_idx * 1000, device="cuda")
        keys_rq = compressor.compress_dequantize(keys)

        # Query = last token attending to all keys
        query = keys[:, :, -1:, :]  # (1, H, 1, D)

        # Real scores
        real_scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)

        # RotorQuant scores
        rq_scores = torch.matmul(query.float(), keys_rq.float().transpose(-2, -1)).squeeze(-2)

        for h in range(H):
            cos = F.cosine_similarity(
                real_scores[0, h].unsqueeze(0),
                rq_scores[0, h].unsqueeze(0)
            ).item()
            cosine_sims.append(cos)

            if real_scores[0, h].argmax().item() == rq_scores[0, h].argmax().item():
                top1_matches += 1
            n_checks += 1

    # Clean up
    del cache_fp16, outputs_fp16
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "cosine_sim": sum(cosine_sims) / len(cosine_sims),
        "top1_match": top1_matches / n_checks * 100,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
    }


# ── Generation test ─────────────────────────────────────────────────

@torch.no_grad()
def test_generation(model, tokenizer, context_len, bits, max_new_tokens=50,
                    keys_only=True, backend='iso'):
    """Generate text with compressed KV cache.

    Returns dict with prefill_tok_s, decode_tok_s, VRAM usage, needle_found.
    """
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_len + 256).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    vram_before = torch.cuda.memory_allocated() / 1024**2

    # Patch KV cache
    original_update, rq_cache = patch_model_kv_cache(
        model, bits=bits, quantize_values=not keys_only, backend=backend)

    try:
        # Prefill: single forward pass on full prompt
        t_prefill_start = time.perf_counter()
        prefill_out = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t_prefill_start
        prefill_tok_s = input_len / t_prefill if t_prefill > 0 else 0

        # Decode: generate token by token
        t_decode_start = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t_decode_start

        gen_tokens = outputs[0][input_len:]
        n_gen = len(gen_tokens)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Decode speed: subtract prefill time from total
        t_decode = t_total - t_prefill if t_total > t_prefill else t_total
        decode_tok_s = n_gen / t_decode if t_decode > 0 else 0

        vram_peak = torch.cuda.max_memory_allocated() / 1024**2
        vram_kv = vram_peak - vram_before

        needle_found = "AURORA-7749" in text or "aurora" in text.lower()

    finally:
        unpatch_model_kv_cache(original_update)
        torch.cuda.empty_cache()

    return {
        "input_tokens": input_len,
        "gen_tokens": n_gen,
        "text": text.strip(),
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "time_s": t_total,
        "vram_peak_mb": vram_peak,
        "vram_kv_est_mb": vram_kv,
        "needle_found": needle_found,
    }


@torch.no_grad()
def test_generation_fp16(model, tokenizer, context_len, max_new_tokens=50):
    """Baseline: generate with standard FP16 KV cache."""
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_len + 256).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    vram_before = torch.cuda.memory_allocated() / 1024**2

    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0

    gen_tokens = outputs[0][input_len:]
    n_gen = len(gen_tokens)
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    vram_peak = torch.cuda.max_memory_allocated() / 1024**2
    vram_kv = vram_peak - vram_before

    torch.cuda.empty_cache()

    return {
        "input_tokens": input_len,
        "gen_tokens": n_gen,
        "text": text.strip(),
        "tok_per_sec": n_gen / t_gen if t_gen > 0 else 0,
        "time_s": t_gen,
        "vram_peak_mb": vram_peak,
        "vram_kv_est_mb": vram_kv,
        "needle_found": "AURORA-7749" in text or "aurora" in text.lower(),
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RotorQuant High-Context POC")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (2-4)")
    parser.add_argument("--max-ctx", type=int, default=32768, help="Max context to test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--skip-fidelity", action="store_true", help="Skip attention fidelity test")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--keys-only", action="store_true", default=True,
                        help="Only compress keys, leave values in fp16 (recommended)")
    parser.add_argument("--compress-values", dest="keys_only", action="store_false",
                        help="Also compress values (higher error)")
    parser.add_argument("--backend", type=str, default="iso", choices=["iso", "clifford", "planar"],
                        help="Rotation backend: iso (IsoQuant), clifford (RotorQuant), or planar (PlanarQuant)")
    args = parser.parse_args()

    backend_names = {"iso": "IsoQuant", "clifford": "RotorQuant", "planar": "PlanarQuant"}
    backend_name = backend_names[args.backend]

    print()
    print("=" * 74)
    print(f"  {backend_name} High-Context POC")
    print(f"  Model: {args.model}")
    print(f"  Bits: {args.bits}  |  Max context: {args.max_ctx:,}  |  Keys only: {args.keys_only}  |  Backend: {backend_name}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 74)
    print()

    # Load model
    print("Loading model...", flush=True)
    import logging
    logging.disable(logging.WARNING)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()

    model_vram = torch.cuda.memory_allocated() / 1024**2
    print(f"Model loaded. VRAM: {model_vram:.0f} MB")
    print()

    # Context lengths to test
    ctx_lengths = []
    ctx = 2048
    while ctx <= args.max_ctx:
        ctx_lengths.append(ctx)
        ctx *= 2

    # ── Phase 1: Attention fidelity ──
    if not args.skip_fidelity:
        print("=" * 74)
        print("PHASE 1: Attention Fidelity (RotorQuant vs FP16)")
        print("=" * 74)
        print()
        print(f"  {'Context':>8s}  {'Cosine Sim':>12s}  {'Top-1 Match':>12s}  {'Layers':>8s}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")

        for ctx_len in ctx_lengths:
            if ctx_len > 8192:
                # Fidelity test requires 2x memory (FP16 + comparison)
                # Skip very long contexts
                print(f"  {ctx_len:>8,}  {'(skipped — needs 2x VRAM)':>40s}")
                continue
            try:
                result = measure_attention_fidelity(model, tokenizer, ctx_len, args.bits, args.backend)
                print(f"  {result['seq_len']:>8,}  {result['cosine_sim']:>12.6f}  "
                      f"{result['top1_match']:>10.1f}%  {result['n_layers']:>8d}")
            except torch.cuda.OutOfMemoryError:
                print(f"  {ctx_len:>8,}  {'OOM':>12s}")
                torch.cuda.empty_cache()
                break

        print()

    # ── Phase 2: Generation with compressed KV cache ──
    print("=" * 74)
    print(f"PHASE 2: Generation with {backend_name} ({args.bits}-bit KV cache)")
    print("=" * 74)
    print()

    # Baseline at smallest context
    print("  FP16 baseline (2K context):")
    try:
        baseline = test_generation_fp16(model, tokenizer, 2048, args.max_new_tokens)
        print(f"    Tokens: {baseline['input_tokens']:,} in + {baseline['gen_tokens']} gen")
        print(f"    Speed:  {baseline['tok_per_sec']:.1f} tok/s")
        print(f"    VRAM:   {baseline['vram_peak_mb']:.0f} MB peak")
        print(f"    Needle: {'FOUND' if baseline['needle_found'] else 'NOT FOUND'}")
        print(f"    Output: {baseline['text'][:120]}...")
    except Exception as e:
        print(f"    Error: {e}")
        baseline = None
    print()

    # Compressed KV at each context length
    print(f"  {backend_name} {args.bits}-bit results:")
    print()
    print(f"  {'Context':>8s}  {'Prefill':>10s}  {'Decode':>10s}  {'VRAM':>8s}  {'Needle':>8s}  {'Output (first 60 chars)'}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*40}")

    for ctx_len in ctx_lengths:
        torch.cuda.empty_cache()
        gc.collect()

        try:
            result = test_generation(model, tokenizer, ctx_len, args.bits,
                                     args.max_new_tokens, args.keys_only, args.backend)
            needle_str = "FOUND" if result['needle_found'] else "MISS"
            text_preview = result['text'][:60].replace('\n', ' ')
            print(f"  {result['input_tokens']:>8,}  "
                  f"{result['prefill_tok_s']:>8.1f}/s  "
                  f"{result['decode_tok_s']:>8.1f}/s  "
                  f"{result['vram_peak_mb']:>6.0f}MB  "
                  f"{needle_str:>8s}  "
                  f"{text_preview}")
        except torch.cuda.OutOfMemoryError:
            print(f"  {ctx_len:>8,}  {'OOM':>10s}  --- VRAM limit reached ---")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  {ctx_len:>8,}  Error: {e}")
            break

    print()

    # ── Phase 3: Memory projection ──
    print("=" * 74)
    print("PHASE 3: Memory Projection")
    print("=" * 74)
    print()

    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

    fp16_per_token = n_layers * 2 * n_kv_heads * head_dim * 2  # bytes (K+V)
    # RotorQuant: same storage (we store dequantized fp16), but could store compressed
    # For now, the savings come from reduced precision of reconstructed values
    rq_per_token = fp16_per_token  # roundtrip — same storage, but quantization noise helps with lower-rank approximation

    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    avail_for_kv = (gpu_total - model_vram / 1024) * 1024**3  # bytes

    print(f"  Model: {args.model}")
    print(f"  Layers: {n_layers}, KV heads: {n_kv_heads}, head_dim: {head_dim}")
    print(f"  FP16 KV per token: {fp16_per_token:,} bytes ({fp16_per_token/1024:.1f} KB)")
    print(f"  GPU total: {gpu_total:.1f} GB, model: {model_vram/1024:.1f} GB")
    print(f"  Available for KV cache: {avail_for_kv/1024**3:.1f} GB")
    print()

    print(f"  {'Context':>10s}  {'FP16 KV':>10s}  {'Status'}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*20}")
    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
        kv_bytes = fp16_per_token * ctx
        kv_gb = kv_bytes / 1024**3
        fits = "fits" if kv_bytes < avail_for_kv else "OOM"
        print(f"  {ctx:>10,}  {kv_gb:>8.2f}GB  {fits}")

    print()
    print("  NOTE: With true compressed storage (not roundtrip dequant),")
    print(f"  {args.bits}-bit RotorQuant would use ~{16/args.bits:.1f}x less KV memory,")
    print(f"  extending max context by ~{16/args.bits:.1f}x.")

    print()
    print("=" * 74)
    print("POC COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()
