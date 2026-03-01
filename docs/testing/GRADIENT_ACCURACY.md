# Full-Sequence K/V Gradient Accumulation

## The Point

Our backward pass only trains K and V weights using feedback from the position that created them — it ignores feedback from every later position that read them.

## The Picture

Imagine a relay race where each runner hands a baton to all future runners. After the race, you only tell each runner how *their own* finish time went — but you never tell them how the baton they passed affected everyone downstream. The early runners never learn that their handoff was sloppy.

## The Proof

In `forward_backward_one()` (line ~2016), when backpropagating through attention:

```c
// K backward for current position: d_k += d_score[TL-1] * q
for (size_t d = 0; d < hd; d++)
    d_k_cur[hoff + d] += d_score_h[TL - 1] * sv_q[L][hoff + d];
```

We only accumulate `d_k_cur` — the gradient for the **current** position's key. But positions `t=0..TL-2` also used their cached keys to influence this position's attention scores. Those earlier keys never receive gradient signal from later positions.

The alternative implementation does it correctly with persistent accumulators:

```c
// Their approach: accumulate K/V gradients across ALL positions
float dk_accum[N_LAYER][BLOCK_SIZE][N_EMBD];  // survives across positions
float dv_accum[N_LAYER][BLOCK_SIZE][N_EMBD];

// During backward at each position:
dk_accum[li][tt][hs + j] += d_al[tt] * act->q[li][hs + j] * scale;
dv_accum[li][tt][hs + j] += act->aw[li][h][tt] * d_ao[hs + j];
```

## The Push

**Don't implement this now.** At our current scale (BLOCK_SIZE=16, N_LAYER=1, N_EMBD=16), the per-position approximation is good enough — early positions have minimal downstream influence in a 16-token window. But **do implement this** if any of these triggers fire:

1. **BLOCK_SIZE > 64** — longer sequences amplify the missing gradient signal
2. **N_LAYER > 2** — deeper models compound the error across layers
3. **Training loss plateaus** — unexplained convergence ceilings could stem from biased K/V gradients
4. **Scaling research** — any experiment where gradient fidelity matters (e.g., comparing training dynamics against PyTorch reference)

---

## Implementation Sketch

If/when this becomes necessary, here's the minimal change:

### New function signature

```c
// Replace per-position forward_backward_one() with:
scalar_t forward_backward_sequence(
    const Model *model,
    const size_t *tokens,      // [n_positions + 1] token IDs
    size_t n_positions,         // number of training positions
    scalar_t **keys,
    scalar_t **values,
    size_t *cache_len,
    scalar_t *grad_buffer
);
```

### Key changes

1. **Forward pass**: loop `pos=0..n-1`, save all activations per position
2. **Backward pass**: loop `pos=n-1..0` in reverse, maintaining `dk_accum[L][t][ne]` and `dv_accum[L][t][ne]` arrays that accumulate across positions
3. **K/V gradient routing**: at position `pos`, backprop K/V gradients to `dk_accum[L][pos]` instead of a temporary `d_k_cur`, then when processing position `pos` backward, use `dk_accum[L][pos]` (which already has contributions from all later positions)

### Memory overhead

```
Per sequence: 2 × N_LAYER × BLOCK_SIZE × N_EMBD × sizeof(scalar_t)
  Current defaults: 2 × 1 × 16 × 16 × 8 = 4 KB   (trivial)
  Scaled (4L/128B/256E): 2 × 4 × 128 × 256 × 8 = 2 MB  (still fine)
```

### API impact

- `forward_backward_one()` remains for compatibility (unchanged approximation)
- New callers use `forward_backward_sequence()` for correct gradients
- `TrainWorker` updated to call the sequence variant

---

## Related Observations from the Comparison

Two other design choices in the alternative implementation worth noting for future reference:

### Squared ReLU activation

They use `x > 0 ? x*x : 0` instead of our `x > 0 ? x : 0`. This produces sparser activations (from the Primer paper) and may improve sample efficiency. The backward derivative is `2x` instead of `1`. Worth benchmarking if MLP capacity becomes a bottleneck.

### Xorshift64 PRNG

Their xorshift64 has better statistical uniformity than our LCG. Not critical for weight init at our scale but worth considering if we ever need high-quality random sampling (e.g., dropout, stochastic depth).
