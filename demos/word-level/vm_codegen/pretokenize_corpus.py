#!/usr/bin/env python3
"""
Pre-tokenize VM DSL corpus for the organelle word-level API.

Reads vm_functions_combined.txt, splits each line using the same
VM-aware scanner logic as vm_scan_token() in main.c, and writes
vm_functions_pretok.txt where all tokens are space-separated.

This lets the generic build_word_vocab() / tokenize_words() (which
split on whitespace) produce the same tokenization as the custom
VM scanner.

Usage:
    python3 pretokenize_corpus.py
"""

import re
import sys
import os

def vm_scan_tokens(line):
    """Python port of vm_scan_token() — splits code into VM DSL tokens."""
    tokens = []
    i = 0
    while i < len(line):
        ch = line[i]

        # Newlines preserved as-is (handled by word vocab as \n token)
        if ch == '\n':
            tokens.append('\n')
            i += 1
            continue

        # Skip spaces (they become whitespace delimiters in output)
        if ch == ' ':
            i += 1
            continue

        # 4-space indent → single token
        if line[i:i+4] == '    ':
            tokens.append('    ')
            i += 4
            continue

        # // comment prefix
        if line[i:i+2] == '//':
            tokens.append('//')
            i += 2
            continue

        # Multi-char operators
        if i + 1 < len(line):
            two = line[i:i+2]
            if two in ('<=', '>=', '!=', '==', '++', '--', '&&', '||'):
                tokens.append(two)
                i += 2
                continue

        # Single punctuation / operator
        if ch in '(){}[];:,+-*/%=<>![]':
            tokens.append(ch)
            i += 1
            continue

        # Number (integer or float)
        if ch.isdigit():
            j = i
            while j < len(line) and (line[j].isdigit() or line[j] == '.'):
                j += 1
            tokens.append(line[i:j])
            i = j
            continue

        # Identifier or keyword
        if ch.isalpha() or ch == '_':
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] == '_'):
                j += 1
            tokens.append(line[i:j])
            i = j
            continue

        # Unknown — skip
        i += 1

    return tokens


def pretokenize_file(input_path, output_path):
    with open(input_path, 'r') as f:
        text = f.read()

    # Split into documents (blank-line separated)
    docs = text.split('\n\n')
    output_docs = []

    for doc in docs:
        doc = doc.strip()
        if not doc:
            continue

        lines = doc.split('\n')
        output_lines = []
        for line in lines:
            tokens = vm_scan_tokens(line)
            if tokens:
                # Join with spaces — this is what tokenize_words() will split on
                output_lines.append(' '.join(tokens))

        if output_lines:
            output_docs.append('\n'.join(output_lines))

    with open(output_path, 'w') as f:
        f.write('\n\n'.join(output_docs) + '\n')

    print(f"Pre-tokenized {len(output_docs)} documents")
    print(f"  Input:  {input_path} ({os.path.getsize(input_path):,} bytes)")
    print(f"  Output: {output_path} ({os.path.getsize(output_path):,} bytes)")

    # Show sample
    if output_docs:
        print(f"\nSample (first doc):")
        for line in output_docs[0].split('\n')[:5]:
            print(f"  {line}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'vm_functions_combined.txt')
    output_path = os.path.join(script_dir, 'vm_functions_pretok.txt')
    pretokenize_file(input_path, output_path)
