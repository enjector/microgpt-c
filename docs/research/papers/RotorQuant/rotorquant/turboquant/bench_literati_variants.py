"""
Quick benchmark of all LiteratiQuant variants on autoregressive PPL.
Tests: basic, v2 (clipping), v3 (iterative), asymmetric, 2bit, rotated, ternary.
"""

import torch, math, time, gc
import torch.nn.functional as F

def make_patcher(compress_fn, device="cuda"):
    """Generic patcher that applies compress_fn to K cache."""
    from transformers import DynamicCache
    prefill_done = {}
    _orig = DynamicCache.update

    def _patch(self, ks, vs, li, ck=None):
        ns = ks.shape[2]
        if ns > 1:
            prefill_done[li] = True
            return _orig(self, ks, vs, li, ck)
        kq = compress_fn(ks, li)
        ko, vo = _orig(self, kq, vs, li, ck)
        ko = ko.clone()
        ko[:, :, -1:, :] = ks
        if prefill_done.get(li) is True:
            ko[:, :, :-1, :] = compress_fn(ko[:, :, :-1, :], li)
            prefill_done[li] = 'done'
        return ko, vo

    return _patch, _orig, prefill_done


@torch.no_grad()
def test_ppl(model, tokenizer, patch_fn, orig_fn, n_tokens=512, prefill_len=256):
    from transformers import DynamicCache
    from datasets import load_dataset

    text = '\n\n'.join(load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    input_ids = tokenizer(text, return_tensors='pt').input_ids[:, :n_tokens].to('cuda')

    DynamicCache.update = patch_fn
    context = input_ids[:, :prefill_len]
    out = model(context, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1:, :]

    nlls = []
    for i in range(input_ids.shape[1] - prefill_len):
        token = input_ids[:, prefill_len + i:prefill_len + i + 1]
        nll = -F.log_softmax(logits, dim=-1)[0, 0, token[0, 0]].item()
        nlls.append(nll)
        mask = torch.ones(1, prefill_len + i + 1, device='cuda', dtype=torch.long)
        out = model(token, attention_mask=mask, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1:, :]

    DynamicCache.update = orig_fn
    ppl = math.exp(sum(nlls) / len(nlls))
    del cache; torch.cuda.empty_cache(); gc.collect()
    return ppl


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from turboquant.literatiquant import (
        quantize_literati, compute_scales_mean_abs,
        quantize_literati_v2, quantize_literati_v3,
        quantize_literati_asymmetric, compute_group_stats,
        quantize_literati_2bit, compute_scales_for_2bit,
    )

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"),
        device_map="auto", dtype=torch.float16)
    model.eval()

    G = 128

    # Define all variants
    variants = {}

    # 1. Basic: sign * mean_abs
    def compress_basic(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        scales = compute_scales_mean_abs(flat, G)
        return quantize_literati(flat, scales, G).to(ks.dtype).reshape(ks.shape)
    variants["1-bit basic"] = compress_basic

    # 2. V2: clipping + optimal scale
    def compress_v2(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        return quantize_literati_v2(flat, G).to(ks.dtype).reshape(ks.shape)
    variants["1-bit v2 (clip)"] = compress_v2

    # 3. V3: iterative
    def compress_v3(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        return quantize_literati_v3(flat, G, n_iter=3).to(ks.dtype).reshape(ks.shape)
    variants["1-bit v3 (iter)"] = compress_v3

    # 4. Asymmetric
    def compress_asym(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        scales, offsets = compute_group_stats(flat, G)
        return quantize_literati_asymmetric(flat, scales, offsets, G).to(ks.dtype).reshape(ks.shape)
    variants["1-bit asymmetric"] = compress_asym

    # 5. 2-bit
    def compress_2bit(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        scales = compute_scales_for_2bit(flat, G)
        return quantize_literati_2bit(flat, scales, G).to(ks.dtype).reshape(ks.shape)
    variants["2-bit uniform"] = compress_2bit

    # 6. Rotated (IsoQuant quaternion + 1-bit)
    from turboquant.literatiquant import LiteratiQuantRotated
    rotated_quants = {}
    def compress_rotated(ks, li):
        D = ks.shape[-1]
        if li not in rotated_quants:
            rotated_quants[li] = LiteratiQuantRotated(D, G, seed=li*1000, device='cuda')
        rq = rotated_quants[li]
        flat = ks.reshape(-1, D).float()
        x_hat, _ = rq(flat)
        return x_hat.to(ks.dtype).reshape(ks.shape)
    variants["1-bit rotated (iso)"] = compress_rotated

    # 7. Ternary {-s, 0, +s} (improved with optimal scale)
    from turboquant.literatiquant import (
        quantize_ternary, quantize_adaptive_clip_median,
        quantize_hybrid_1_2bit, quantize_per_head_norm,
        quantize_temporal_smooth, _temporal_scales,
    )

    def compress_ternary(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        return quantize_ternary(flat, G, zero_thresh=0.3).to(ks.dtype).reshape(ks.shape)
    variants["1.5-bit ternary"] = compress_ternary

    # 8. Adaptive clipping + median scale
    def compress_adapt_median(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        return quantize_adaptive_clip_median(flat, G, percentile=0.95).to(ks.dtype).reshape(ks.shape)
    variants["1-bit adapt+median"] = compress_adapt_median

    # 9. Hybrid 1/2-bit per group
    def compress_hybrid(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        return quantize_hybrid_1_2bit(flat, G, var_percentile=0.7).to(ks.dtype).reshape(ks.shape)
    variants["hybrid 1/2-bit"] = compress_hybrid

    # 10. Per-head normalization
    def compress_perhead(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        return quantize_per_head_norm(flat, G).to(ks.dtype).reshape(ks.shape)
    variants["1-bit per-head norm"] = compress_perhead

    # 11. Temporal smoothing
    _temporal_scales.clear()
    def compress_temporal(ks, li):
        flat = ks.reshape(-1, ks.shape[-1]).float()
        return quantize_temporal_smooth(flat, li, G, alpha=0.3).to(ks.dtype).reshape(ks.shape)
    variants["1-bit temporal smooth"] = compress_temporal

    # 12. Rotation + ternary (combine best ideas)
    def compress_rot_ternary(ks, li):
        D = ks.shape[-1]
        if li not in rotated_quants:
            rotated_quants[li] = LiteratiQuantRotated(D, G, seed=li*1000, device='cuda')
        rq = rotated_quants[li]
        flat = ks.reshape(-1, D).float()
        # Rotate
        norms = torch.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = flat / norms
        x_rot = rq._rotate(x_unit)
        # Ternary on rotated
        x_q_rot = quantize_ternary(x_rot, G, zero_thresh=0.3)
        # Inverse rotate
        x_q = rq._unrotate(x_q_rot)
        return (x_q * norms).to(ks.dtype).reshape(ks.shape)
    variants["1.5-bit rot+ternary"] = compress_rot_ternary

    # 13. Rotation + 2-bit (best quality attempt)
    def compress_rot_2bit(ks, li):
        D = ks.shape[-1]
        if li not in rotated_quants:
            rotated_quants[li] = LiteratiQuantRotated(D, G, seed=li*1000, device='cuda')
        rq = rotated_quants[li]
        flat = ks.reshape(-1, D).float()
        norms = torch.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = flat / norms
        x_rot = rq._rotate(x_unit)
        scales = compute_scales_for_2bit(x_rot, G)
        x_q_rot = quantize_literati_2bit(x_rot, scales, G)
        x_q = rq._unrotate(x_q_rot)
        return (x_q * norms).to(ks.dtype).reshape(ks.shape)
    variants["2-bit rotated"] = compress_rot_2bit

    # 14. RaBitQ with planar rotation
    from turboquant.rabitq import RaBitQ
    rabitq_planar = {}
    def compress_rabitq_planar(ks, li):
        D = ks.shape[-1]
        if li not in rabitq_planar:
            rabitq_planar[li] = RaBitQ(D, rotation='planar', seed=li*1000, device='cuda')
        rq = rabitq_planar[li]
        flat = ks.reshape(-1, D).float()
        x_hat, _ = rq(flat)
        return x_hat.to(ks.dtype).reshape(ks.shape)
    variants["RaBitQ planar"] = compress_rabitq_planar

    # 15. RaBitQ with iso rotation
    rabitq_iso = {}
    def compress_rabitq_iso(ks, li):
        D = ks.shape[-1]
        if li not in rabitq_iso:
            rabitq_iso[li] = RaBitQ(D, rotation='iso', seed=li*1000, device='cuda')
        rq = rabitq_iso[li]
        flat = ks.reshape(-1, D).float()
        x_hat, _ = rq(flat)
        return x_hat.to(ks.dtype).reshape(ks.shape)
    variants["RaBitQ iso"] = compress_rabitq_iso

    # 16. RaBitQ with full random orthogonal
    rabitq_full = {}
    def compress_rabitq_full(ks, li):
        D = ks.shape[-1]
        if li not in rabitq_full:
            rabitq_full[li] = RaBitQ(D, rotation='full', seed=li*1000, device='cuda')
        rq = rabitq_full[li]
        flat = ks.reshape(-1, D).float()
        x_hat, _ = rq(flat)
        return x_hat.to(ks.dtype).reshape(ks.shape)
    variants["RaBitQ full ortho"] = compress_rabitq_full

    # Run all
    print(f"\n{'Variant':>25}  {'PPL':>8}  {'bits/elem':>10}  {'compress':>10}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*10}  {'─'*10}")

    # FP16 baseline
    from transformers import DynamicCache
    orig = DynamicCache.update
    def noop_patch(self, ks, vs, li, ck=None):
        return orig(self, ks, vs, li, ck)
    ppl_fp16 = test_ppl(model, tokenizer, noop_patch, orig)
    print(f"  {'FP16':>25}  {ppl_fp16:>8.2f}  {'16.0':>10}  {'1.0x':>10}")

    for name, cfn in variants.items():
        torch.cuda.empty_cache(); gc.collect()
        patch, orig_fn, _ = make_patcher(cfn)
        try:
            ppl = test_ppl(model, tokenizer, patch, orig_fn)
            bits = "1.125" if "1-bit" in name else "1.5" if "ternary" in name else "2.125" if "2-bit" in name else "?"
            comp = f"{16.0/float(bits):.1f}x" if bits != "?" else "?"
            print(f"  {name:>25}  {ppl:>8.2f}  {bits:>10}  {comp:>10}")
        except Exception as e:
            print(f"  {name:>25}  {'ERROR':>8}  {str(e)[:40]}")


if __name__ == "__main__":
    main()
