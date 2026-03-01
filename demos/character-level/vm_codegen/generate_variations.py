#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""Generate prompt variations per function in vm_functions.txt.

Reads the corpus, extracts each // comment + function body pair,
generates alternative comment variations, and outputs the expanded corpus.
"""

import re, sys, random

# ── Domain-specific synonym dictionaries ──────────────────────────────────

VERB_SYNONYMS = [
    ("compute", "calculate", "evaluate", "find", "determine"),
    ("check", "test", "verify", "validate"),
    ("convert", "transform", "translate"),
    ("apply", "perform", "execute", "run"),
    ("count", "tally", "enumerate"),
    ("sum", "total", "add up", "accumulate"),
]

NOUN_SYNONYMS = [
    ("factorial", "n factorial", "product of integers"),
    ("fibonacci", "fib number", "fibonacci sequence value"),
    ("absolute value", "abs value", "magnitude"),
    ("maximum", "max", "largest", "greatest"),
    ("minimum", "min", "smallest", "least"),
    ("average", "mean", "arithmetic mean"),
    ("interest", "accrued interest", "earned interest"),
    ("distance", "separation", "gap"),
    ("area", "surface area"),
    ("perimeter", "boundary length"),
    ("volume", "capacity"),
    ("energy", "kinetic energy"),
    ("velocity", "speed"),
    ("percentage", "percent", "fraction"),
    ("midpoint", "center point", "halfway point"),
    ("reciprocal", "inverse", "one over"),
    ("sigmoid", "logistic function"),
    ("relu", "rectified linear"),
    ("gradient", "derivative", "slope"),
    ("precision", "positive predictive value"),
    ("recall", "sensitivity", "true positive rate"),
    ("radius", "r"),
    ("height", "h"),
    ("weight", "mass"),
    ("temperature", "temp"),
    ("circle", "circular shape"),
    ("triangle", "triangular shape"),
    ("rectangle", "rectangular shape"),
    ("sphere", "ball"),
    ("cylinder", "cylindrical shape"),
    ("cube root", "cubic root"),
    ("square root", "sqrt"),
    ("hypotenuse", "longest side"),
    ("bmi", "body mass index"),
    ("tax", "taxation"),
    ("salary", "pay", "wages"),
    ("tip", "gratuity"),
    ("discount", "reduction", "markdown"),
    ("profit", "gain", "earnings"),
    ("loss", "deficit"),
    ("density", "mass density"),
    ("pressure", "force per area"),
    ("momentum", "linear momentum"),
    ("impulse", "force impulse"),
    ("wavelength", "wave length"),
    ("frequency", "rate of oscillation"),
    ("prime", "prime number"),
    ("palindrome", "symmetric number"),
    ("digits", "digit count"),
]


# ── Variation generation strategies ───────────────────────────────────────

def make_variations(comment_text):
    """Generate alternative comments from an original // comment string."""
    inner = comment_text.strip()
    if inner.startswith("//"):
        inner = inner[2:].strip()

    variations = []
    lower = inner.lower()

    # Strategy 1: Replace action verb with synonyms
    has_verb = any(lower.startswith(v) for vg in VERB_SYNONYMS for v in vg)
    if has_verb:
        for vgroup in VERB_SYNONYMS:
            for v in vgroup:
                if lower.startswith(v + " "):
                    rest = inner[len(v)+1:]
                    others = [x for x in vgroup if x != v]
                    for alt in others[:2]:
                        variations.append(f"// {alt} {rest}")
                    break
            if variations:
                break

    if not has_verb:
        variations.append(f"// compute {inner}")
        variations.append(f"// calculate {inner}")

    # Strategy 2: Rephrase using noun synonyms
    for ngroup in NOUN_SYNONYMS:
        for n in ngroup:
            if n in lower:
                others = [x for x in ngroup if x != n and x not in lower]
                if others:
                    alt = inner.replace(n, others[0], 1)
                    if alt != inner:
                        variations.append(f"// {alt}")
                    if len(others) > 1:
                        alt2 = inner.replace(n, others[1], 1)
                        if alt2 != inner:
                            variations.append(f"// {alt2}")
                break

    # Strategy 3: Add "of value" / "for input" suffix
    if not any(x in lower for x in ["of value", "for input", "of number"]):
        variations.append(f"// {inner} of value")

    # Strategy 4: Simplified short form
    words = inner.split()
    if len(words) > 3:
        short = " ".join(words[:3])
        variations.append(f"// {short}")
    elif len(words) <= 3:
        variations.append(f"// {inner} function")

    # Strategy 5: Technical alternative
    variations.append(f"// {inner} implementation")

    # Deduplicate and take exactly 6
    seen = {comment_text.strip()}
    unique = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    # Pad with mechanical variants if needed
    fillers = [
        f"// {inner} routine",
        f"// {inner} computation",
        f"// perform {inner}",
        f"// evaluate {inner}",
        f"// {inner} method",
        f"// {inner} operation",
        f"// get {inner}",
        f"// return {inner}",
    ]
    for fl in fillers:
        if len(unique) >= 6:
            break
        if fl not in seen:
            seen.add(fl)
            unique.append(fl)

    return unique[:6]


def parse_corpus(text):
    """Parse corpus into list of (comment, body) pairs."""
    blocks = re.split(r'\n\n+', text.strip())
    functions = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Find the // comment on the first line
        m = re.match(r'(//.*?)\n(.*)', block, re.DOTALL)
        if m:
            comment = m.group(1)
            body = m.group(2)
            functions.append((comment, body))
    return functions


def main():
    # Process both original and phase4 corpus files
    input_files = ["vm_functions.txt"]
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]
    else:
        # Auto-detect phase4 file
        import os
        if os.path.exists("vm_functions_phase4.txt"):
            input_files.append("vm_functions_phase4.txt")

    output_file = "vm_functions_variations.txt"

    all_functions = []
    for input_file in input_files:
        with open(input_file, 'r') as f:
            text = f.read()
        functions = parse_corpus(text)
        print(f"Parsed {len(functions)} functions from {input_file}")
        all_functions.extend(functions)

    print(f"Total base functions: {len(all_functions)}")

    # Track which (comment, body) pairs we already have
    existing_comments = set()
    for comment, body in all_functions:
        existing_comments.add(comment.strip())

    # Generate variations
    new_entries = []
    for comment, body in all_functions:
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

    # Merge all into combined
    combined_file = "vm_functions_combined.txt"
    with open(combined_file, 'w') as f:
        for input_file in input_files:
            with open(input_file, 'r') as src:
                f.write(src.read())
                f.write("\n")
        with open(output_file, 'r') as src:
            f.write(src.read())

    # Count combined
    import re
    with open(combined_file, 'r') as f:
        combined_text = f.read()
    combined_funcs = parse_corpus(combined_text)
    print(f"\nCombined corpus: {len(combined_funcs)} total functions")
    print(f"Wrote to {combined_file}")


if __name__ == "__main__":
    main()
