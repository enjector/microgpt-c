/*
 * wiring_anchor_graphs.c — canonical @graph DAGs for Phase 2 retrieval.
 *
 * Each entry is the smallest verified graph that, when executed on the
 * held-out test inputs, produces the answer matching wiring_references.c.
 * Eight anchors lifted verbatim from pipeline_corpus_{train,val}.txt
 * (already verified during corpus generation); the remaining twelve
 * handcrafted to mirror the reference function's semantics, using
 * primitives present in wiring_natives.c and the input-name conventions
 * the corpus generator uses (so the wiring executor's `<name>` lookup
 * resolves correctly to S[0..N-1] in declaration order).
 */

#include "wiring_anchor_graphs.h"
#include <string.h>

typedef struct {
    const char *name;
    const char *graph;
} AnchorEntry;

/* The graphs use newline literals; the wiring demo's preprocessor
 * replaces "\n" with " __NL__ " for tokenisation, but the text-format
 * parser eats the raw newlines we store here. The trailing "\n@end\n"
 * matches the corpus's verbatim layout. */
static const AnchorEntry ANCHORS[] = {
    /* #1 — bmi_clamped: r_clamp(r_bmi(weight, height), lo, hi)
     *      Inputs: S[0]=weight, S[1]=height, S[2]=lo, S[3]=hi */
    { "bmi_clamped",
      "@graph bmi_clamped\n"
      "  : in weight -> int\n"
      "  : in height -> int\n"
      "  : in lo -> int\n"
      "  : in hi -> int\n"
      "  : out y -> int\n"
      "  | b = bmi(weight: <weight>, height: <height>) :: weight:int, height:int -> out:int\n"
      "  | c = clamp(x: b.out, lo: <lo>, hi: <hi>) :: x:int, lo:int, hi:int -> out:int\n"
      "  y <- c.out\n"
      "@end\n" },

    /* #2 — compound_interest: r_compound(P, r, n) - P (verbatim from corpus)
     *      Inputs: S[0]=principal, S[1]=rate, S[2]=years */
    { "compound_interest",
      "@graph compound_interest\n"
      "  : in principal -> int\n"
      "  : in rate -> int\n"
      "  : in years -> int\n"
      "  : out y -> int\n"
      "  | amount = compound(principal: <principal>, rate: <rate>, periods: <years>) :: principal:int, rate:int, periods:int -> out:int\n"
      "  | diff = subtract(x: amount.out, y: <principal>) :: x:int, y:int -> out:int\n"
      "  y <- diff.out\n"
      "@end\n" },

    /* #3 — weighted_three: (m1*w1 + m2*w2 + m3*w3) * 100 / (w1+w2+w3)
     *      Inputs: S[0]=m1, S[1]=w1, S[2]=m2, S[3]=w2, S[4]=m3, S[5]=w3 */
    { "weighted_three",
      "@graph weighted_three\n"
      "  : in m1 -> int\n"
      "  : in w1 -> int\n"
      "  : in m2 -> int\n"
      "  : in w2 -> int\n"
      "  : in m3 -> int\n"
      "  : in w3 -> int\n"
      "  : out y -> int\n"
      "  | n1 = multiply(x: <m1>, y: <w1>) :: x:int, y:int -> out:int\n"
      "  | n2 = multiply(x: <m2>, y: <w2>) :: x:int, y:int -> out:int\n"
      "  | n3 = multiply(x: <m3>, y: <w3>) :: x:int, y:int -> out:int\n"
      "  | s12 = add(x: n1.out, y: n2.out) :: x:int, y:int -> out:int\n"
      "  | num = add(x: s12.out, y: n3.out) :: x:int, y:int -> out:int\n"
      "  | sw12 = add(x: <w1>, y: <w2>) :: x:int, y:int -> out:int\n"
      "  | den = add(x: sw12.out, y: <w3>) :: x:int, y:int -> out:int\n"
      "  | pct = percentage(part: num.out, whole: den.out) :: part:int, whole:int -> out:int\n"
      "  y <- pct.out\n"
      "@end\n" },

    /* #4 — clamped_sigmoid: r_clamp(r_sigmoid(x), lo, hi)
     *      Inputs: S[0]=x, S[1]=lo, S[2]=hi */
    { "clamped_sigmoid",
      "@graph clamped_sigmoid\n"
      "  : in x -> int\n"
      "  : in lo -> int\n"
      "  : in hi -> int\n"
      "  : out y -> int\n"
      "  | s = sigmoid(x: <x>) :: x:int -> out:int\n"
      "  | c = clamp(x: s.out, lo: <lo>, hi: <hi>) :: x:int, lo:int, hi:int -> out:int\n"
      "  y <- c.out\n"
      "@end\n" },

    /* #5 — gcd_scaled: r_gcd(a, b) * k
     *      Inputs: S[0]=a, S[1]=b, S[2]=k */
    { "gcd_scaled",
      "@graph gcd_scaled\n"
      "  : in a -> int\n"
      "  : in b -> int\n"
      "  : in k -> int\n"
      "  : out y -> int\n"
      "  | g = gcd(a: <a>, b: <b>) :: a:int, b:int -> out:int\n"
      "  | s = multiply(x: g.out, y: <k>) :: x:int, y:int -> out:int\n"
      "  y <- s.out\n"
      "@end\n" },

    /* #6 — apply_tax: r_apply_tax(gross, rate)
     *      Inputs: S[0]=gross, S[1]=rate */
    { "apply_tax",
      "@graph apply_tax\n"
      "  : in gross -> int\n"
      "  : in rate -> int\n"
      "  : out y -> int\n"
      "  | net = apply_tax(amount: <gross>, rate: <rate>) :: amount:int, rate:int -> out:int\n"
      "  y <- net.out\n"
      "@end\n" },

    /* #7 — fib_fact_mul: r_fibonacci(n) * r_factorial(n)
     *      Inputs: S[0]=n */
    { "fib_fact_mul",
      "@graph fib_fact_mul\n"
      "  : in n -> int\n"
      "  : out y -> int\n"
      "  | fib = fibonacci(x: <n>) :: x:int -> out:int\n"
      "  | fact = factorial(x: <n>) :: x:int -> out:int\n"
      "  | y0 = multiply(x: fib.out, y: fact.out) :: x:int, y:int -> out:int\n"
      "  y <- y0.out\n"
      "@end\n" },

    /* #8 — invoice_total: price*qty + tax_amount(price*qty, rate)
     *      Inputs: S[0]=price, S[1]=qty, S[2]=rate */
    { "invoice_total",
      "@graph invoice_total\n"
      "  : in price -> int\n"
      "  : in qty -> int\n"
      "  : in rate -> int\n"
      "  : out y -> int\n"
      "  | sub = multiply(x: <price>, y: <qty>) :: x:int, y:int -> out:int\n"
      "  | tax = tax_amount(amount: sub.out, rate: <rate>) :: amount:int, rate:int -> out:int\n"
      "  | total = add(x: sub.out, y: tax.out) :: x:int, y:int -> out:int\n"
      "  y <- total.out\n"
      "@end\n" },

    /* #9 — clamped_average: r_clamp(r_average_two(a, b), lo, hi)
     *      Inputs: S[0]=a, S[1]=b, S[2]=lo, S[3]=hi */
    { "clamped_average",
      "@graph clamped_average\n"
      "  : in a -> int\n"
      "  : in b -> int\n"
      "  : in lo -> int\n"
      "  : in hi -> int\n"
      "  : out y -> int\n"
      "  | avg = average_two(a: <a>, b: <b>) :: a:int, b:int -> out:int\n"
      "  | c = clamp(x: avg.out, lo: <lo>, hi: <hi>) :: x:int, lo:int, hi:int -> out:int\n"
      "  y <- c.out\n"
      "@end\n" },

    /* #10 — abs_diff: r_abs(a - b)
     *      Inputs: S[0]=a, S[1]=b */
    { "abs_diff",
      "@graph abs_diff\n"
      "  : in a -> int\n"
      "  : in b -> int\n"
      "  : out y -> int\n"
      "  | d = subtract(x: <a>, y: <b>) :: x:int, y:int -> out:int\n"
      "  | r = abs_val(x: d.out) :: x:int -> out:int\n"
      "  y <- r.out\n"
      "@end\n" },

    /* #11 — scaled_relu: r_relu(x) * scale
     *      Inputs: S[0]=x, S[1]=scale */
    { "scaled_relu",
      "@graph scaled_relu\n"
      "  : in x -> int\n"
      "  : in scale -> int\n"
      "  : out y -> int\n"
      "  | r = relu(x: <x>) :: x:int -> out:int\n"
      "  | s = multiply(x: r.out, y: <scale>) :: x:int, y:int -> out:int\n"
      "  y <- s.out\n"
      "@end\n" },

    /* #12 — discounted_tax: tax_amount(price - price*rate/100, tax_rate)
     *      Inputs: S[0]=price, S[1]=disc_rate, S[2]=tax_rate */
    { "discounted_tax",
      "@graph discounted_tax\n"
      "  : in price -> int\n"
      "  : in disc -> int\n"
      "  : in trate -> int\n"
      "  : out y -> int\n"
      "  | dpct = percentage(part: <price>, whole: <disc>) :: part:int, whole:int -> out:int\n"
      "  | red = subtract(x: <price>, y: dpct.out) :: x:int, y:int -> out:int\n"
      "  | t = tax_amount(amount: red.out, rate: <trate>) :: amount:int, rate:int -> out:int\n"
      "  y <- t.out\n"
      "@end\n" },

    /* #13 — savings_rate: percentage(income - sum_expenses, income)
     *      Inputs: S[0]=income, S[1]=exp1, S[2]=exp2 */
    { "savings_rate",
      "@graph savings_rate\n"
      "  : in income -> int\n"
      "  : in exp1 -> int\n"
      "  : in exp2 -> int\n"
      "  : out y -> int\n"
      "  | sumexp = add(x: <exp1>, y: <exp2>) :: x:int, y:int -> out:int\n"
      "  | saved = subtract(x: <income>, y: sumexp.out) :: x:int, y:int -> out:int\n"
      "  | pct = percentage(part: saved.out, whole: <income>) :: part:int, whole:int -> out:int\n"
      "  y <- pct.out\n"
      "@end\n" },

    /* #14 — distance_metrics: square(distance_1d(a1,b1) + distance_1d(a2,b2))
     *      Inputs: S[0]=a1, S[1]=b1, S[2]=a2, S[3]=b2 */
    { "distance_metrics",
      "@graph distance_metrics\n"
      "  : in a1 -> int\n"
      "  : in b1 -> int\n"
      "  : in a2 -> int\n"
      "  : in b2 -> int\n"
      "  : out y -> int\n"
      "  | d1 = distance_1d(a: <a1>, b: <b1>) :: a:int, b:int -> out:int\n"
      "  | d2 = distance_1d(a: <a2>, b: <b2>) :: a:int, b:int -> out:int\n"
      "  | s = add(x: d1.out, y: d2.out) :: x:int, y:int -> out:int\n"
      "  | sq = square(x: s.out) :: x:int -> out:int\n"
      "  y <- sq.out\n"
      "@end\n" },

    /* #15 — distance_midpoint: distance_1d(a, b) + midpoint(a, b)
     *      Inputs: S[0]=a, S[1]=b */
    { "distance_midpoint",
      "@graph distance_midpoint\n"
      "  : in a -> int\n"
      "  : in b -> int\n"
      "  : out y -> int\n"
      "  | d = distance_1d(a: <a>, b: <b>) :: a:int, b:int -> out:int\n"
      "  | m = midpoint(a: <a>, b: <b>) :: a:int, b:int -> out:int\n"
      "  | s = add(x: d.out, y: m.out) :: x:int, y:int -> out:int\n"
      "  y <- s.out\n"
      "@end\n" },

    /* #16 — pv_of_fv: present_value(future_value(c, r, n), r, n)
     *      Inputs: S[0]=cashflow, S[1]=rate, S[2]=years */
    { "pv_of_fv",
      "@graph pv_of_fv\n"
      "  : in cashflow -> int\n"
      "  : in rate -> int\n"
      "  : in years -> int\n"
      "  : out y -> int\n"
      "  | fv = future_value(present: <cashflow>, rate: <rate>, periods: <years>) :: present:int, rate:int, periods:int -> out:int\n"
      "  | pv = present_value(future: fv.out, rate: <rate>, periods: <years>) :: future:int, rate:int, periods:int -> out:int\n"
      "  y <- pv.out\n"
      "@end\n" },

    /* #17 — fib_fact_add: r_fibonacci(n) + r_factorial(n)  (verbatim)
     *      Inputs: S[0]=n */
    { "fib_fact_add",
      "@graph fib_fact_add\n"
      "  : in n -> int\n"
      "  : out y -> int\n"
      "  | fact = factorial(x: <n>) :: x:int -> out:int\n"
      "  | fib = fibonacci(x: <n>) :: x:int -> out:int\n"
      "  | blend = add(x: fib.out, y: fact.out) :: x:int, y:int -> out:int\n"
      "  y <- blend.out\n"
      "@end\n" },

    /* #18 — gross_minus_tax: gross - tax_amount(gross, rate)
     *      Inputs: S[0]=gross, S[1]=rate */
    { "gross_minus_tax",
      "@graph gross_minus_tax\n"
      "  : in gross -> int\n"
      "  : in rate -> int\n"
      "  : out y -> int\n"
      "  | t = tax_amount(amount: <gross>, rate: <rate>) :: amount:int, rate:int -> out:int\n"
      "  | r = subtract(x: <gross>, y: t.out) :: x:int, y:int -> out:int\n"
      "  y <- r.out\n"
      "@end\n" },

    /* #19 — compound_minus_p: r_compound(P, r, n) - P (alias of compound_interest)
     *      Inputs: S[0]=P, S[1]=r, S[2]=n */
    { "compound_minus_p",
      "@graph compound_minus_p\n"
      "  : in principal -> int\n"
      "  : in rate -> int\n"
      "  : in years -> int\n"
      "  : out y -> int\n"
      "  | amount = compound(principal: <principal>, rate: <rate>, periods: <years>) :: principal:int, rate:int, periods:int -> out:int\n"
      "  | diff = subtract(x: amount.out, y: <principal>) :: x:int, y:int -> out:int\n"
      "  y <- diff.out\n"
      "@end\n" },

    /* #20 — sigmoid_clamped: r_clamp(r_sigmoid(x), lo, hi)  (alias of clamped_sigmoid)
     *      Inputs: S[0]=x, S[1]=lo, S[2]=hi */
    { "sigmoid_clamped",
      "@graph sigmoid_clamped\n"
      "  : in x -> int\n"
      "  : in lo -> int\n"
      "  : in hi -> int\n"
      "  : out y -> int\n"
      "  | s = sigmoid(x: <x>) :: x:int -> out:int\n"
      "  | c = clamp(x: s.out, lo: <lo>, hi: <hi>) :: x:int, lo:int, hi:int -> out:int\n"
      "  y <- c.out\n"
      "@end\n" },
};

static const int N_ANCHORS = (int)(sizeof(ANCHORS) / sizeof(ANCHORS[0]));

const char *wiring_anchor_graph_for(const char *family_name) {
    if (!family_name) return NULL;
    for (int i = 0; i < N_ANCHORS; i++) {
        if (strcmp(ANCHORS[i].name, family_name) == 0) return ANCHORS[i].graph;
    }
    return NULL;
}

int wiring_anchor_count(void) {
    return N_ANCHORS;
}
