#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""Generate prompt variations for c_wiring.txt.

Reads the wiring corpus, extracts each /* comment */ + function body pair,
generates 4 additional comment variations, and outputs the expanded corpus.
"""

import re, sys

# ── Wiring-specific synonym dictionaries ──────────────────────────────────

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
]

NOUN_SYNONYMS = [
    ("mean", "average", "arithmetic mean"),
    ("standard deviation", "stddev", "std dev"),
    ("variance", "squared deviation"),
    ("median", "middle value"),
    ("array", "vector", "buffer", "data"),
    ("signal", "waveform", "series"),
    ("element", "value", "entry"),
    ("threshold", "cutoff", "limit"),
    ("range", "span", "extent"),
    ("constant", "scalar", "factor"),
    ("ratio", "proportion", "relative measure"),
    ("residual", "remainder", "error"),
    ("gradient", "derivative", "slope"),
    ("output", "result", "destination"),
    ("coefficient", "weight", "parameter"),
]


def make_variations(comment_text):
    """Generate 4 alternative comments from an original."""
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
        f"/* {inner} computation */",
    ]
    for f in fillers:
        if len(unique) >= 4:
            break
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique[:4]


def parse_corpus(text):
    """Parse corpus into list of (comment, body) pairs."""
    blocks = re.split(r'\n\n+', text.strip())
    functions = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.match(r'(/\*.*?\*/)\s*\n(.*)', block, re.DOTALL)
        if m:
            comment = m.group(1)
            body = m.group(2)
            functions.append((comment, body))
    return functions


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "c_wiring.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_file, 'r') as f:
        text = f.read()

    functions = parse_corpus(text)
    print(f"Parsed {len(functions)} functions from {input_file}")

    existing_comments = set()
    for comment, body in functions:
        existing_comments.add(comment.strip())

    new_entries = []
    for comment, body in functions:
        variations = make_variations(comment)
        for var_comment in variations:
            if var_comment.strip() not in existing_comments:
                existing_comments.add(var_comment.strip())
                new_entries.append((var_comment, body))

    print(f"Generated {len(new_entries)} new variation entries")

    if output_file:
        # Write only variations
        with open(output_file, 'w') as f:
            for comment, body in new_entries:
                f.write(f"\n{comment}\n{body}\n")
        print(f"Wrote variations to {output_file}")
    else:
        # Append to input file
        with open(input_file, 'a') as f:
            for comment, body in new_entries:
                f.write(f"\n{comment}\n{body}\n")
        print(f"Appended {len(new_entries)} variations to {input_file}")

    total_docs = len(functions) + len(new_entries)
    print(f"Total documents: {total_docs}")


if __name__ == "__main__":
    main()
