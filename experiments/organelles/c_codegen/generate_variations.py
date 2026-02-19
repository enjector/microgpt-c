#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""Generate 5 prompt variations per function in c_functions.txt.

Reads the corpus, extracts each /* comment */ + function body pair,
generates 4 additional comment variations, and outputs the expanded corpus.
"""

import re, sys, random

# ── Domain-specific synonym dictionaries ──────────────────────────────────

VERB_SYNONYMS = [
    ("compute", "calculate", "evaluate", "find", "determine"),
    ("sort", "order", "arrange", "rank"),
    ("apply", "perform", "execute", "run"),
    ("filter", "smooth", "process"),
    ("normalize", "normalise", "rescale", "standardize"),
    ("estimate", "approximate", "predict"),
    ("generate", "produce", "create"),
]

NOUN_SYNONYMS = [
    ("mean", "average", "arithmetic mean"),
    ("standard deviation", "stddev", "std dev", "volatility of values"),
    ("variance", "squared deviation"),
    ("median", "middle value"),
    ("dot product", "inner product", "scalar product"),
    ("distance", "separation", "difference measure"),
    ("correlation", "linear association"),
    ("covariance", "joint variation"),
    ("volatility", "dispersion", "variability"),
    ("return", "yield", "gain"),
    ("price", "value", "fair value"),
    ("probability", "likelihood", "chance"),
    ("energy", "work capacity"),
    ("force", "interaction strength"),
    ("velocity", "speed"),
    ("acceleration", "rate of velocity change"),
    ("wavelength", "wave period"),
    ("frequency", "oscillation rate"),
    ("entropy", "disorder measure", "information content"),
    ("window", "taper", "windowing function"),
    ("spectrum", "frequency content", "spectral density"),
    ("ratio", "proportion", "relative measure"),
]

DOMAIN_PREFIXES = {
    "finance": ["financial ", "portfolio ", "risk ", "trading "],
    "physics": ["physical ", "classical ", ""],
    "quantum": ["quantum ", "qubit ", ""],
    "signal": ["signal ", "DSP ", "digital "],
    "stats": ["statistical ", "sample ", ""],
}

# ── Variation generation strategies ───────────────────────────────────────

def make_variations(comment_text):
    """Generate 4 alternative comments from an original comment string."""
    # Strip /* */ markers
    inner = comment_text.strip()
    if inner.startswith("/*"):
        inner = inner[2:]
    if inner.endswith("*/"):
        inner = inner[:-2]
    inner = inner.strip()
    
    variations = []
    lower = inner.lower()
    
    # Strategy 1: Add/change action verb prefix
    has_verb = any(lower.startswith(v) for vg in VERB_SYNONYMS for v in vg)
    if has_verb:
        # Replace the verb with alternatives
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
    
    if not has_verb:
        # Add verb prefixes
        variations.append(f"/* compute {inner} */")
        variations.append(f"/* calculate {inner} */")
    
    # Strategy 2: Rephrase using noun synonyms
    for ngroup in NOUN_SYNONYMS:
        for n in ngroup:
            if n in lower:
                others = [x for x in ngroup if x != n and x not in lower]
                if others:
                    alt = inner.replace(n, others[0], 1)
                    if alt != inner:
                        variations.append(f"/* {alt} */")
                    if len(others) > 1:
                        alt2 = inner.replace(n, others[1], 1)
                        if alt2 != inner:
                            variations.append(f"/* {alt2} */")
                break
    
    # Strategy 3: Add "of array" / "for series" / "of data" suffix
    if not any(x in lower for x in ["of array", "for series", "of data", "from"]):
        variations.append(f"/* {inner} of array */")
    
    # Strategy 4: Simplified short form
    words = inner.split()
    if len(words) > 3:
        short = " ".join(words[:3])
        variations.append(f"/* {short} */")
    elif len(words) <= 3:
        variations.append(f"/* {inner} function */")
    
    # Strategy 5: Technical alternative
    variations.append(f"/* {inner} implementation */")
    
    # Deduplicate and take exactly 4
    seen = {comment_text.strip()}
    unique = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    
    # Pad with mechanical variants if needed
    fillers = [
        f"/* {inner} routine */",
        f"/* {inner} computation */",
        f"/* perform {inner} */",
        f"/* evaluate {inner} */",
        f"/* {inner} method */",
        f"/* {inner} algorithm */",
        f"/* {inner} operation */",
        f"/* {inner} formula */",
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
    # Split on blank lines
    blocks = re.split(r'\n\n+', text.strip())
    functions = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Find the comment
        m = re.match(r'(/\*.*?\*/)\s*\n(.*)', block, re.DOTALL)
        if m:
            comment = m.group(1)
            body = m.group(2)
            functions.append((comment, body))
    return functions


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "c_functions.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "c_functions_variations.txt"
    
    with open(input_file, 'r') as f:
        text = f.read()
    
    functions = parse_corpus(text)
    print(f"Parsed {len(functions)} functions from {input_file}")
    
    # Track which (comment, body) pairs we already have
    existing_comments = set()
    for comment, body in functions:
        existing_comments.add(comment.strip())
    
    # Generate variations
    new_entries = []
    for comment, body in functions:
        variations = make_variations(comment)
        for var_comment in variations:
            if var_comment.strip() not in existing_comments:
                existing_comments.add(var_comment.strip())
                new_entries.append((var_comment, body))
    
    print(f"Generated {len(new_entries)} new variation entries")
    
    # Write variations to output file
    with open(output_file, 'w') as f:
        for comment, body in new_entries:
            f.write(f"\n{comment}\n{body}\n")
    
    print(f"Wrote variations to {output_file}")
    
    # Show stats
    total_docs = len(functions) + len(new_entries)
    print(f"Total documents after merge: {total_docs}")


if __name__ == "__main__":
    main()
