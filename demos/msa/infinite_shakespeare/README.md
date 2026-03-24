# Memory Sparse Attention: Infinite Shakespeare

This demo modifies the standard word-level Shakespeare generation pipeline by integrating **Memory Sparse Attention (MSA)** to break the fundamental Discretisation Wall (context limit) of the underlying transformer.

By default, this model features a microscopic context window (`BLOCK_SIZE = 64`). Under standard generation, inference would fail or require a full $O(L^2)$ matrix recalculation via a sliding window upon reaching word 65. 

### Mechanism of Action
Instead of crashing or dropping context, this pipeline utilizes an `MsaPool` to capture infinitely scaling associative arrays:
1. When token 64 is reached, the oldest 32 tokens in the KV cache are mathematically averaged and compressed into a fixed-scale Latent Memory Vector using `msa_pool_chunk`.
2. The active context window slides down by 32 positions (virtually instant `memmove`).
3. For the newly opened slot `pos = 0`, the router executes a Cosine Similarity sweep (`msa_route_top_1`) across all historical chunks the model has *ever* generated over its lifetime. 
4. The most semantically relevant historical context vector is dynamically injected into the active KV cache (`msa_expand_context`).

### Performance Telemetry: Baseline vs. MSA Infinite

When both the baseline `w_shakespeare` generator and the new `msa_infinite_shakespeare` generator are targeted to generate **500 words**, the Discretisation Wall perfectly demonstrates the architectural divide.

#### 1. Baseline Shakespeare (`demos/word-level/shakespeare`)
```text
[sample 1 — seed: "the"]
the trotting entreat wolves friday! or lartius shed rom grip'd grass madding through suffered scandaliz'd lines principal they lingers spongy rack hermione'o trust double jig affected persuade opens iachimo resolv'd tides wakes solemn lips looks craves spies patience linger itch silvius clouds aufidius tapers dismal'st gods resolution request distrain'd throwing something stabs swords rest treasons bid italian somerset envious far controversy
  >> 62 words in 0.002s (34261 tok/s)
```
**Result:** The baseline engine completely stalled. Its active context window filled up, and generation was forcefully aborted at precisely 62 words (64 tokens) to prevent a memory violation crash.

#### 2. MSA Infinite Shakespeare (`demos/msa/infinite_shakespeare`)
```text
[sample 1 — seed: "the"]
the trotting entreat wolves friday! or lartius shed rom grip'd grass madding through suffered scandaliz'd lines principal they lingers spongy rack hermione'o trust double jig affected persuade opens iachimo resolv'd tides wakes solemn lips looks craves spies patience linger itch silvius clouds aufidius tapers dismal'st gods resolution request distrain'd throwing something stabs swords rest treasons bid italian somerset envious far controversy banquo there's slanderer notice sum foul treachery scarce quondam' noise westmoreland pays adding makes doomsday expedition running disturb'd inc norfolk abroad skill anne antenor air performance delay ... (truncated for brevity) ... lost hour ingratitude caps do iachimo seest she'll offend profound din uncle conduit feast robes norfolk
  >> 500 words in 0.015s (33734 tok/s)
```
**Result:** The MSA engine effortlessly bypassed the 64-token wall natively. Whenever it reached token 64, it shifted its context and pooled the forgotten memory into the `MsaPool`. It reached the 500-word target maintaining an incredible `~33,000 tok/s` — experiencing only a negligible `~1.5%` throughput latency hit to afford its `O(N)` algorithmic Cosine similarity sweeps!
### Running the Generator
```bash
# Build the application
cmake --build build --target msa_infinite_shakespeare

# Execute the binary from the build directory
cd build
./msa_infinite_shakespeare
```
