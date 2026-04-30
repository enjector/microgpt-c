#!/usr/bin/env bash
# tools/scaling_leakage_audit.sh — three-axis leakage audit for the
# scaling-curve experiment. Invokes from build/.
#
# Audit A: verbatim — held-out prompts that appear word-for-word in the train corpus.
# Audit B: near-duplicate — bag-of-words Jaccard ≥ 0.7 against any training prompt.
# Audit C: lexical anchoring — for each held-out prompt, how many of its content
#          words appear ONLY in its own family's training. High → TF-IDF wins on
#          single-word distinctness, inflating the apparent 1:1 result.
#
# Usage: cd build && bash ../tools/scaling_leakage_audit.sh

set -u
HELDOUT="pipeline_corpus_scaling_heldout.txt"
TRAIN="pipeline_corpus_phase4_train.txt"

for f in "$HELDOUT" "$TRAIN"; do
    [[ -e "$f" ]] || { echo "ERROR: $f not found in $(pwd)" >&2; exit 1; }
done

# Extract held-out as "<family>\t<prompt>" pairs.
HELDOUT_TSV=$(mktemp)
awk '
    /^# REFERENCE:/ { fam = $3; next }
    /^\/\/ /        { sub(/^\/\/ +/, ""); print fam "\t" tolower($0) }
' "$HELDOUT" > "$HELDOUT_TSV"

# Extract train as "<family>\t<prompt>" pairs.
# corpus_expand format: "// prompt\n... @graph family\n..."
# Parse the family from the @graph line that immediately follows.
TRAIN_TSV=$(mktemp)
awk '
    /^\/\/ / {
        sub(/^\/\/ +/, "")
        prompt = tolower($0)
        next
    }
    /^@graph / {
        fam = $2
        # corpus_expand emits family names like "circle_area_ratio_op_3" — strip the _op_<n> suffix.
        sub(/_op_[0-9]+$/, "", fam)
        if (prompt != "") { print fam "\t" prompt; prompt = "" }
    }
' "$TRAIN" > "$TRAIN_TSV"

n_heldout=$(wc -l < "$HELDOUT_TSV" | tr -d ' ')
n_train=$(wc -l < "$TRAIN_TSV" | tr -d ' ')

echo "============================================================"
echo "Scaling-curve leakage audit"
echo "============================================================"
echo "Held-out prompts: $n_heldout"
echo "Train prompts:    $n_train"
echo ""

# ============================================================
# Audit A: verbatim
# ============================================================
echo "--- Audit A: verbatim leakage ---"
verbatim_hits=0
while IFS=$'\t' read -r fam prompt; do
    if awk -F'\t' -v p="$prompt" '$2 == p { found=1; exit } END { exit !found }' "$TRAIN_TSV"; then
        echo "  LEAK[verbatim]: $fam  →  $prompt"
        verbatim_hits=$((verbatim_hits + 1))
    fi
done < "$HELDOUT_TSV"
echo "Audit A total: $verbatim_hits / $n_heldout verbatim leaks"
echo ""

# ============================================================
# Audit B: near-duplicate (Jaccard ≥ 0.7) — inverted-index version
# ============================================================
# Algorithm: for each held-out prompt, store its word-set. Build an
# inverted index from word → list of held-out prompt indices. Then
# stream the train file: for each train prompt, tokenize once and look
# up each word in the inverted index — for any held-out prompt that
# shares ≥1 word, increment its intersection counter. After processing
# the train prompt, compute Jaccard for each held-out prompt that
# accumulated any intersection. Total work: O(train_tokens + intersections).
echo "--- Audit B: near-duplicate (Jaccard) leakage ---"
echo "For each held-out prompt, max Jaccard against any training prompt:"
awk -F'\t' '
    function tokenize_to(line, target_arr,    n, words, i) {
        n = split(line, words, /[^a-z]+/)
        delete target_arr
        for (i = 1; i <= n; i++) if (words[i] != "" && length(words[i]) >= 2) target_arr[words[i]] = 1
    }
    NR == FNR {
        hold_fam[NR] = $1
        hold[NR] = $2
        n_hold = NR
        # Pre-tokenize and store word-set for each held-out prompt.
        tokenize_to($2, hold_words)
        sz = 0
        for (w in hold_words) {
            sz++
            # Inverted index: word → space-sep list of held-out indices.
            inv_idx[w] = (w in inv_idx ? inv_idx[w] " " NR : NR)
        }
        hold_size[NR] = sz
        next
    }
    {
        # Train side: tokenize, count intersection with each held-out prompt
        # that shares any word.
        tokenize_to($2, train_words)
        delete isect_count
        train_size = 0
        for (w in train_words) {
            train_size++
            if (w in inv_idx) {
                n = split(inv_idx[w], hold_idxs, " ")
                for (i = 1; i <= n; i++) isect_count[hold_idxs[i]]++
            }
        }
        for (idx in isect_count) {
            isect = isect_count[idx]
            union = hold_size[idx] + train_size - isect
            j = (union > 0) ? (isect / union) : 0
            if (j > best[idx]) {
                best[idx] = j
                best_fam[idx] = $1
                best_prompt[idx] = $2
            }
        }
    }
    END {
        n_concerning = 0
        for (i = 1; i <= n_hold; i++) {
            tag = (best[i] >= 0.7) ? "WARN" : (best[i] >= 0.5 ? "med " : "ok  ")
            if (best[i] >= 0.7) n_concerning++
            printf "  [%s] %.3f  %s ← train:%s  \"%s\"\n",
                tag, best[i], hold_fam[i], best_fam[i], best_prompt[i]
        }
        printf "Audit B total: %d / %d held-out prompts with max-Jaccard ≥ 0.7\n",
            n_concerning, n_hold
    }
' "$HELDOUT_TSV" "$TRAIN_TSV"
echo ""

# ============================================================
# Audit C: lexical anchoring
# ============================================================
echo "--- Audit C: lexical anchoring (vocabulary exclusivity) ---"
echo "For each held-out prompt, how many of its content words appear in"
echo "ONLY its own family's training data (= 'lexical anchors')."
awk -F'\t' '
    function tokenize(s, out,    n, w, words, i) {
        n = split(s, words, /[^a-z]+/)
        delete out
        for (i = 1; i <= n; i++) if (words[i] != "" && length(words[i]) >= 3) out[words[i]] = 1
    }
    NR == FNR {
        # Hold-out side
        hold_fam[NR] = $1
        hold[NR] = $2
        n_hold = NR
        next
    }
    {
        # Train side: record (word, family) presence.
        tokenize($2, words)
        for (w in words) {
            if (!(w in word_seen) || word_seen[w] != $1) {
                if (w in word_seen) word_multi[w] = 1
                word_seen[w] = $1
            }
        }
    }
    END {
        for (i = 1; i <= n_hold; i++) {
            tokenize(hold[i], words)
            n_words = 0; n_anchor = 0; n_unseen = 0
            anchor_list = ""
            for (w in words) {
                n_words++
                if (!(w in word_seen)) {
                    n_unseen++
                } else if (!(w in word_multi) && word_seen[w] == hold_fam[i]) {
                    n_anchor++
                    anchor_list = anchor_list (anchor_list ? "," : "") w
                }
            }
            anchor_pct = (n_words > 0) ? 100.0 * n_anchor / n_words : 0
            tag = (anchor_pct >= 50) ? "HIGH" : (anchor_pct >= 25 ? "med " : "low ")
            printf "  [%s] %3.0f%% anchors (%d/%d, %d unseen)  %s  anchors=[%s]\n",
                tag, anchor_pct, n_anchor, n_words, n_unseen, hold_fam[i], anchor_list
        }
    }
' "$HELDOUT_TSV" "$TRAIN_TSV"
echo ""

rm -f "$HELDOUT_TSV" "$TRAIN_TSV"
echo "============================================================"
echo "Audit complete. Three signals to interpret:"
echo "  Audit A > 0  → strict verbatim leakage (FORBIDDEN[] failed)"
echo "  Audit B ≥ 0.7 → near-duplicate; TF-IDF can win on overlap alone"
echo "  Audit C ≥ 50% anchors → TF-IDF wins via single-word distinctness"
echo "============================================================"
