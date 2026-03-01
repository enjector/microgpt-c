#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""Generate flat-string corpora for the OPA code composition pipeline.

Reads c_wiring.txt, extracts each /* comment */ + function body pair,
identifies the helper functions called, and outputs:
  - c_planner.txt:  /* comment */ → seq|fn1|fn2|...
  - c_judge.txt:    seq|fn1|fn2 → PASS / FAIL
  - c_registry.txt: function name → return type (for validation)
  - test_intents.txt: held-out test prompts (20%)

Usage:
    python3 generate_corpus.py ../c_wiringgen/c_wiring.txt ../c_codegen/c_functions.txt
"""

import re
import sys
import random
import os

random.seed(42)

# ── Synonym dictionaries for comment variation ────────────────────────

VERB_SYNONYMS = [
    ("normalize", "normalise", "rescale", "standardize"),
    ("compute", "calculate", "evaluate", "find", "determine"),
    ("smooth", "filter", "process"),
    ("apply", "perform", "execute", "run"),
    ("chain", "pipe", "cascade", "connect"),
    ("blend", "mix", "combine", "merge"),
    ("sort", "order", "arrange", "rank"),
    ("fill", "populate", "generate", "create"),
    ("extract", "select", "pick", "pull"),
    ("accumulate", "aggregate", "reduce", "collect"),
    ("subtract", "remove", "take away"),
    ("multiply", "scale", "amplify"),
    ("differentiate", "derive", "gradient of"),
    ("integrate", "sum", "accumulate"),
    ("downsample", "decimate", "reduce rate"),
    ("upsample", "interpolate", "increase rate"),
]

NOUN_SYNONYMS = [
    ("mean", "average", "arithmetic mean"),
    ("standard deviation", "stddev", "std dev"),
    ("variance", "squared deviation"),
    ("median", "middle value"),
    ("array", "vector", "buffer", "data", "signal"),
    ("signal", "waveform", "series", "stream"),
    ("element", "value", "entry", "item"),
    ("threshold", "cutoff", "limit"),
    ("range", "span", "extent"),
    ("constant", "scalar", "factor"),
    ("ratio", "proportion", "relative measure"),
    ("residual", "remainder", "error"),
    ("gradient", "derivative", "slope"),
    ("output", "result", "destination"),
    ("coefficient", "weight", "parameter"),
    ("frequency", "spectral", "harmonic"),
    ("noise", "jitter", "perturbation"),
    ("envelope", "amplitude", "magnitude"),
]


def make_comment_variations(comment_text):
    """Generate up to 4 alternative comments from an original."""
    inner = comment_text.strip()
    if inner.startswith("/*"):
        inner = inner[2:]
    if inner.endswith("*/"):
        inner = inner[:-2]
    inner = inner.strip()

    variations = []
    lower = inner.lower()

    # Strategy 1: Verb substitution
    for vgroup in VERB_SYNONYMS:
        for v in vgroup:
            if lower.startswith(v + " "):
                rest = inner[len(v)+1:]
                others = [x for x in vgroup if x != v]
                for alt in others[:2]:
                    variations.append(f"/* {alt} {rest} */")
                break
        if variations:
            break

    if not variations:
        variations.append(f"/* compute {inner} */")
        variations.append(f"/* calculate {inner} */")

    # Strategy 2: Noun synonyms
    for ngroup in NOUN_SYNONYMS:
        for n in ngroup:
            if n in lower:
                others = [x for x in ngroup if x != n and x not in lower]
                if others:
                    alt = inner.replace(n, others[0], 1)
                    if alt != inner:
                        variations.append(f"/* {alt} */")
                break

    # Strategy 3: Suffix additions
    if not any(x in lower for x in ["of array", "of data", "of signal"]):
        variations.append(f"/* {inner} of array */")

    # Strategy 4: Simplified short form
    words = inner.split()
    if len(words) > 3:
        short = " ".join(words[:3])
        variations.append(f"/* {short} */")
    else:
        variations.append(f"/* {inner} function */")

    # Deduplicate and take exactly 4
    seen = {comment_text.strip()}
    unique = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    fillers = [
        f"/* {inner} operation */",
        f"/* perform {inner} */",
        f"/* {inner} routine */",
    ]
    for filler in fillers:
        if len(unique) >= 4:
            break
        if filler not in seen:
            seen.add(filler)
            unique.append(filler)

    return unique[:4]

# ── Known function names from c_codegen (primitives) ──────────────────

def extract_primitives(codegen_path):
    """Extract function name → return type from c_functions.txt."""
    registry = {}
    with open(codegen_path) as f:
        for line in f:
            m = re.match(r'^(void|double|int)\s+(\w+)\s*\(', line)
            if m:
                ret_type = m.group(1)
                name = m.group(2)
                registry[name] = ret_type
    return registry


def extract_called_functions(body, all_known):
    """Extract function calls from a wiring body, in order of appearance."""
    # Match function calls: word followed by (
    calls = re.findall(r'\b([a-z_][a-z_0-9]*)\s*\(', body)
    # Filter to known functions (exclude C stdlib, loops, etc.)
    skip = {'for', 'if', 'while', 'return', 'sizeof', 'printf', 'malloc',
            'calloc', 'free', 'fabs', 'sqrt', 'exp', 'log', 'pow', 'sin',
            'cos', 'tan', 'atan2', 'ceil', 'floor', 'rand', 'abs', 'memcpy',
            'memset', 'memmove', 'strlen', 'strcmp', 'strncmp', 'snprintf',
            'fopen', 'fclose', 'fprintf', 'fscanf', 'fread', 'fwrite'}
    # Keep only known functions OR functions that look like our helpers
    result = []
    seen = set()
    for c in calls:
        if c in skip:
            continue
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def parse_wiring_corpus(wiring_path):
    """Parse c_wiring.txt into (comment, fn_name, body) triples."""
    with open(wiring_path) as f:
        text = f.read()

    blocks = re.split(r'\n\n+', text.strip())
    entries = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.match(r'(/\*.*?\*/)\s*\n(.*)', block, re.DOTALL)
        if m:
            comment = m.group(1)
            body = m.group(2)
            # Extract the defined function name
            fn_m = re.match(r'(?:void|double|int)\s+(\w+)\s*\(', body)
            fn_name = fn_m.group(1) if fn_m else None
            entries.append((comment, fn_name, body))
    return entries


def generate_planner_corpus(entries, all_known):
    """Generate comment → seq|fn1|fn2 training pairs."""
    pairs = []
    for comment, fn_name, body in entries:
        called = extract_called_functions(body, all_known)
        # Exclude the wrapper function itself from the call list
        if fn_name:
            called = [c for c in called if c != fn_name]
        if not called:
            continue
        flat_str = "seq|" + "|".join(called)
        pairs.append((comment, flat_str))
    return pairs


def generate_judge_corpus(planner_pairs, all_known):
    """Generate PASS/FAIL validation pairs."""
    all_fn_names = list(all_known.keys())
    pairs = []

    # PASS examples: all valid planner outputs
    for comment, flat_str in planner_pairs:
        pairs.append((flat_str, "PASS"))

    # FAIL examples: synthetic bad plans
    for comment, flat_str in planner_pairs:
        fns = flat_str.split("|")[1:]  # skip "seq"

        # Type 1: Insert a nonexistent function name
        bad_name = random.choice(fns) + "_zzz"
        bad_fns = list(fns)
        idx = random.randint(0, len(bad_fns))
        bad_fns.insert(idx, bad_name)
        pairs.append(("seq|" + "|".join(bad_fns), "FAIL"))

        # Type 2: Empty plan
        if random.random() < 0.3:
            pairs.append(("seq|", "FAIL"))

        # Type 3: Single random token (not a function)
        if random.random() < 0.3:
            pairs.append(("seq|12345", "FAIL"))

        # Type 4: Shuffle to create unlikely ordering (less reliable fail)
        if len(fns) >= 3 and random.random() < 0.5:
            shuffled = list(fns)
            random.shuffle(shuffled)
            if shuffled != fns:
                pairs.append(("seq|" + "|".join(shuffled), "FAIL"))

    random.shuffle(pairs)
    return pairs


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 generate_corpus.py <c_wiring.txt> <c_functions.txt>")
        sys.exit(1)

    wiring_path = sys.argv[1]
    codegen_path = sys.argv[2]

    # Extract primitive function registry
    registry = extract_primitives(codegen_path)
    print(f"Registry: {len(registry)} primitive functions")

    # Also add wiring-defined functions as "known"
    entries = parse_wiring_corpus(wiring_path)
    print(f"Wiring corpus: {len(entries)} compositions")

    # Build full known-function set (registry + wiring-defined)
    all_known = dict(registry)
    for comment, fn_name, body in entries:
        if fn_name and fn_name not in all_known:
            # Infer return type from signature
            m = re.match(r'(void|double|int)', body.strip())
            if m:
                all_known[fn_name] = m.group(1)
    print(f"Total known functions: {len(all_known)}")

    # Generate planner corpus
    planner_pairs = generate_planner_corpus(entries, all_known)
    print(f"Planner pairs: {len(planner_pairs)}")

    # Train/test split (80/20) — split BEFORE variation expansion
    random.shuffle(planner_pairs)
    split = int(0.8 * len(planner_pairs))
    train_pairs = planner_pairs[:split]
    test_pairs = planner_pairs[split:]
    print(f"Train (base): {len(train_pairs)}, Test: {len(test_pairs)}")

    # Expand training set with comment variations
    existing_comments = set(c for c, _ in train_pairs)
    variation_pairs = []
    for comment, flat_str in train_pairs:
        variations = make_comment_variations(comment)
        for var_comment in variations:
            if var_comment not in existing_comments:
                existing_comments.add(var_comment)
                variation_pairs.append((var_comment, flat_str))
    train_pairs = train_pairs + variation_pairs
    random.shuffle(train_pairs)
    print(f"Train (expanded): {len(train_pairs)} (+{len(variation_pairs)} variations)")

    # Generate judge corpus (from train set only)
    judge_pairs = generate_judge_corpus(train_pairs, all_known)
    print(f"Judge pairs: {len(judge_pairs)} ({sum(1 for _,v in judge_pairs if v=='PASS')} PASS, "
          f"{sum(1 for _,v in judge_pairs if v=='FAIL')} FAIL)")

    # Write output files
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # c_planner.txt: one entry per double-newline block
    # Format: comment\nseq|fn1|fn2
    with open(os.path.join(out_dir, "c_planner.txt"), "w") as f:
        for comment, flat_str in train_pairs:
            f.write(f"{comment}\n{flat_str}\n\n")
    print(f"Wrote c_planner.txt ({len(train_pairs)} entries)")

    # c_judge.txt: one entry per block
    # Format: seq|fn1|fn2\nPASS  or  seq|fn1|fn2\nFAIL
    with open(os.path.join(out_dir, "c_judge.txt"), "w") as f:
        for flat_str, verdict in judge_pairs:
            f.write(f"{flat_str}\n{verdict}\n\n")
    print(f"Wrote c_judge.txt ({len(judge_pairs)} entries)")

    # test_intents.txt: held-out test prompts with expected plans
    with open(os.path.join(out_dir, "test_intents.txt"), "w") as f:
        for comment, flat_str in test_pairs:
            f.write(f"{comment}\n{flat_str}\n\n")
    print(f"Wrote test_intents.txt ({len(test_pairs)} entries)")

    # c_registry.txt: function name → type (for validation at runtime)
    with open(os.path.join(out_dir, "c_registry.txt"), "w") as f:
        for name in sorted(all_known.keys()):
            f.write(f"{name}|{all_known[name]}\n")
    print(f"Wrote c_registry.txt ({len(all_known)} entries)")

    # Print some examples
    print("\n=== Example planner entries ===")
    for comment, flat_str in train_pairs[:5]:
        print(f"  {comment}")
        print(f"  → {flat_str}")
        print()

    # Stats
    all_fns_in_plans = set()
    chain_lengths = []
    for _, flat_str in planner_pairs:
        fns = flat_str.split("|")[1:]
        all_fns_in_plans.update(fns)
        chain_lengths.append(len(fns))

    print(f"\n=== Corpus Statistics ===")
    print(f"Unique functions referenced: {len(all_fns_in_plans)}")
    print(f"Chain length: min={min(chain_lengths)}, max={max(chain_lengths)}, "
          f"avg={sum(chain_lengths)/len(chain_lengths):.1f}")
    print(f"Planner vocab size: ~{len(set('|'.join(f for _,f in planner_pairs)))}")


if __name__ == "__main__":
    main()
