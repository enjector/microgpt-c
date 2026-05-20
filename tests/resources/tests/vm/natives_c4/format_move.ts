// E08 Phase 2 — OUTPUT_BEHAVIOUR shape for Connect-4.
// The host stages the model's next-token output via current_move_handle;
// the behaviour returns the parsed column digit (0..6) or -1 on parse fail.

declare function c4_parse_token(): number;

function eval(): number {
    var col = c4_parse_token();
    return col;
}
