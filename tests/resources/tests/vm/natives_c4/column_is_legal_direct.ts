// Same shape as declare_function_calculation1.ts — single return.
declare function c4_column_is_legal_n(col: number): number;

function eval(col: number): number {
    var r = c4_column_is_legal_n(col);
    return r;
}
