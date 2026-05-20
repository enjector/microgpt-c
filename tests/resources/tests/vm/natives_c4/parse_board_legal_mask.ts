// E08 Phase 2 — exercises the c4_legal_column_mask extern.
// The host has set current_board_handle to point at a 42-char Connect-4
// board.  The behaviour returns the 7-bit mask of legal column drops.

declare function c4_legal_column_mask(): number;

function eval(): number {
    var mask = c4_legal_column_mask();
    return mask;
}
