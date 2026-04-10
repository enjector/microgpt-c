"""
Per-layer codebook calibration for RotorQuant.

Collects actual post-rotor component statistics from a calibration forward pass,
then fits Lloyd-Max centroids per layer to the REAL distribution instead of the
theoretical Gaussian approximation.

Usage:
    from turboquant.calibrate import calibrate_rotorquant, CalibratedRotorQuantCompressor
    codebooks = calibrate_rotorquant(model, tokenizer, bits=3, n_tokens=2048)
    compressor = CalibratedRotorQuantCompressor(codebooks, bits=3)
"""

import torch
import numpy as np
import math
from typing import Optional

from .rotorquant import RotorQuantMSE
from .triton_kernels import triton_rotor_sandwich, pack_rotors_for_triton
from .clifford import E1, E2, E3, E123


def _fit_centroids_1d(samples: np.ndarray, n_centroids: int) -> np.ndarray:
    """Fit 1D Lloyd-Max centroids via iterative k-means on scalar samples."""
    if len(samples) < n_centroids:
        return np.linspace(samples.min(), samples.max(), n_centroids)

    # Initialize from quantiles (better than random for 1D)
    quantiles = np.linspace(0, 1, n_centroids + 2)[1:-1]
    centroids = np.quantile(samples, quantiles)

    # Lloyd's algorithm (1D k-means)
    for _ in range(100):
        # Boundaries = midpoints between consecutive centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2

        # Assign each sample to nearest centroid
        bins = np.searchsorted(boundaries, samples)

        # Update centroids as mean of assigned samples
        new_centroids = np.zeros(n_centroids)
        for i in range(n_centroids):
            mask = bins == i
            if mask.any():
                new_centroids[i] = samples[mask].mean()
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    return np.sort(centroids)


@torch.no_grad()
def calibrate_rotorquant(
    model,
    tokenizer,
    bits: int = 3,
    n_tokens: int = 2048,
    device: str = "cuda",
) -> dict:
    """Collect post-rotor statistics and fit per-layer centroids.

    Args:
        model: HuggingFace model (already loaded)
        tokenizer: tokenizer for calibration text
        bits: quantization bit width
        n_tokens: calibration tokens from wikitext-2
        device: cuda device

    Returns:
        dict mapping layer_idx → {'vector': centroids_tensor, 'trivector': centroids_tensor,
                                   'rotors': packed_rotors_tensor}
    """
    from transformers import DynamicCache
    from datasets import load_dataset

    print("  Loading calibration data...", flush=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens].to(device)

    config = model.config
    text_config = getattr(config, 'text_config', config)
    head_dim = getattr(text_config, 'head_dim',
                       text_config.hidden_size // text_config.num_attention_heads)
    n_layers = text_config.num_hidden_layers

    n_groups = (head_dim + 2) // 3
    n_centroids = 2 ** bits

    # Create per-layer rotors (same as RotorQuantMSE would)
    per_layer_rotors = {}
    for li in range(n_layers):
        rq = RotorQuantMSE(head_dim, bits, seed=li * 1000, device=device)
        per_layer_rotors[li] = pack_rotors_for_triton(rq.rotors).to(device)

    # Collect post-rotor statistics by hooking into cache update
    stats = {li: {'v1': [], 'v2': [], 'v3': [], 't7': []} for li in range(n_layers)}

    _orig = DynamicCache.update

    def _collection_hook(self, key_states, value_states, layer_idx, cache_kwargs=None):
        D = key_states.shape[-1]
        flat = key_states.reshape(-1, D).float()

        # Normalize
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        flat_unit = flat / norms

        # Rotor sandwich
        pk = per_layer_rotors[layer_idx]
        mv_rot = triton_rotor_sandwich(flat_unit, pk)

        # Collect component statistics
        stats[layer_idx]['v1'].append(mv_rot[:, :, E1].cpu().numpy().ravel())
        stats[layer_idx]['v2'].append(mv_rot[:, :, E2].cpu().numpy().ravel())
        stats[layer_idx]['v3'].append(mv_rot[:, :, E3].cpu().numpy().ravel())
        stats[layer_idx]['t7'].append(mv_rot[:, :, E123].cpu().numpy().ravel())

        return _orig(self, key_states, value_states, layer_idx, cache_kwargs)

    print("  Running calibration forward pass...", flush=True)
    DynamicCache.update = _collection_hook
    model(input_ids, use_cache=True)
    DynamicCache.update = _orig
    torch.cuda.empty_cache()

    # Fit per-layer centroids
    print(f"  Fitting {n_layers} × 4 codebooks ({n_centroids} centroids each)...", flush=True)
    codebooks = {}

    for li in range(n_layers):
        # Concatenate all collected samples
        v1 = np.concatenate(stats[li]['v1'])
        v2 = np.concatenate(stats[li]['v2'])
        v3 = np.concatenate(stats[li]['v3'])
        t7 = np.concatenate(stats[li]['t7'])

        # Vector grades: fit shared codebook from all 3 components (same distribution)
        all_vector = np.concatenate([v1, v2, v3])
        vector_centroids = _fit_centroids_1d(all_vector, n_centroids)

        # Trivector: fit separate codebook
        trivector_centroids = _fit_centroids_1d(t7, n_centroids)

        codebooks[li] = {
            'vector': torch.tensor(vector_centroids, dtype=torch.float32, device=device),
            'trivector': torch.tensor(trivector_centroids, dtype=torch.float32, device=device),
            'rotors': per_layer_rotors[li],
        }

    # Print summary
    print(f"  Calibration complete. Sample stats (layer 0 vs layer 35):")
    for li in [0, n_layers - 1]:
        v = np.concatenate(stats[li]['v1'])
        t = np.concatenate(stats[li]['t7'])
        uncalib_range = codebooks[li]['vector'][[0, -1]].tolist()
        print(f"    Layer {li:>2d}: vector std={v.std():.4f} range=[{v.min():.4f},{v.max():.4f}]"
              f" → centroids [{uncalib_range[0]:.4f},{uncalib_range[1]:.4f}]"
              f"  trivector std={t.std():.4f}")

    return codebooks


class CalibratedRotorQuantCompressor:
    """Per-layer key compressor using calibrated codebooks."""

    def __init__(self, codebooks: dict, bits: int, device: str = "cuda"):
        self.codebooks = codebooks
        self.bits = bits
        self.device = device

    @torch.no_grad()
    def compress_dequantize(self, key_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Quantize → dequantize with per-layer calibrated codebook."""
        B, H, S, D = key_states.shape
        flat = key_states.reshape(-1, D).float()

        cb = self.codebooks[layer_idx]
        pk = cb['rotors']
        c_v = cb['vector']
        c_t = cb['trivector']

        # Normalize
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        flat_unit = flat / norms

        # Rotor sandwich
        mv_rot = triton_rotor_sandwich(flat_unit, pk)

        # Quantize with calibrated centroids
        b_v = (c_v[:-1] + c_v[1:]) / 2
        b_t = (c_t[:-1] + c_t[1:]) / 2

        v1 = mv_rot[:, :, E1]; v2 = mv_rot[:, :, E2]
        v3 = mv_rot[:, :, E3]; t7 = mv_rot[:, :, E123]

        # Nearest centroid
        q_v1 = c_v[torch.searchsorted(b_v, v1.contiguous())]
        q_v2 = c_v[torch.searchsorted(b_v, v2.contiguous())]
        q_v3 = c_v[torch.searchsorted(b_v, v3.contiguous())]
        q_t7 = c_t[torch.searchsorted(b_t, t7.contiguous())]

        # Reconstruct MV
        mv_q = torch.zeros_like(mv_rot)
        mv_q[:, :, E1] = q_v1; mv_q[:, :, E2] = q_v2
        mv_q[:, :, E3] = q_v3; mv_q[:, :, E123] = q_t7

        # Inverse sandwich
        from .triton_kernels import triton_rotor_inverse_sandwich
        k_mse = triton_rotor_inverse_sandwich(mv_q, pk, D)

        # Rescale
        k_mse = k_mse * norms
        return k_mse.to(key_states.dtype).reshape(B, H, S, D)
