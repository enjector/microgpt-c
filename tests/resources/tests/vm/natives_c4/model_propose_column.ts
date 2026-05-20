// E11 — c4_model_propose_column extern.
//
// Asks the host (via the runtime-installed callback) to run a real
// model-driven column proposal against the currently-staged board.
// The argument is base temperature × 100 (clamped 1..100 by the host).
//
// The behaviour returns the proposed column 0..6 verbatim, or -1 if
// either no callback is installed (test harness without an organelle) or
// the model produced an unparseable / illegal proposal.  The OQL runtime
// adapter wraps this return as a one-hot legal mask via `1 << col` when
// the column is legal, falling back to the full legal mask otherwise.

declare function c4_model_propose_column(t: number): number;

function eval(): number {
    var col = c4_model_propose_column(20);
    return col;
}
