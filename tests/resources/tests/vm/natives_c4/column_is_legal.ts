// E08 Phase 2 — VALIDATE_BEHAVIOUR shape, no-arg form.
// The behaviour parses the currently-staged move token, then checks legality.
// Returns 1 if legal, 0 if illegal (or parse failed).
//
// NOTE on VM return semantics:  the existing VM treats `return` as a value-
// push only — it does NOT actually exit the function at runtime.  Multiple
// `return X` statements all execute in source order and the last value
// pushed wins.  Therefore every BEHAVIOUR body assigns to a single result
// variable and returns it once at the end (conditions3-shape pattern).

declare function c4_parse_token(): number;
declare function c4_column_is_legal_n(col: number): number;

function eval(): number {
    var col = c4_parse_token();
    var result = 0;
    if (col >= 0) {
        result = c4_column_is_legal_n(col);
    }
    return result;
}
