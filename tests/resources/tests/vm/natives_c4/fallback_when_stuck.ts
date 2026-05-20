// E08 Phase 2 — FALLBACK_BEHAVIOUR shape for Connect-4.
// Returns the centre column (3) when entropy is high; -1 otherwise.

declare function c4_last_entropy(): number;
declare function c4_centre_col(): number;
declare function c4_column_is_legal_n(col: number): number;

// Single-return idiom — see header note in column_is_legal.ts.
function eval(): number {
    var ent = c4_last_entropy();
    var result = -1;
    if (ent > 0.8) {
        var cc = c4_centre_col();
        var legal = c4_column_is_legal_n(cc);
        if (legal > 0) {
            result = cc;
        }
    }
    return result;
}
