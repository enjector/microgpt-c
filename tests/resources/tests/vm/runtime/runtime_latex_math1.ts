a1() = (123 + 1 + (2 * 2))

a(x) = x + 1

b(x, y) = x + y;

c(x, y) = x ^ 2 + y ^ 3

d(\alpha) = \alpha + 1

e() =\sum_{ i = 0 }^ 20 i + 3;

f() =\sum_{ i = 0 }^ 20(i + 3) + i

g() =\sum_{ i = 0 }^ 10 \sum_{ j = 0 }^ 5 i + j

h(n) =\sum_{ i = 0 }^ n \sum_{ j = 0 }^ n i + j

k(n) =\sum_{ i = 0 }^ 10 \sum_{ j = 0 }^ (n + 1) i + j

m(x, n) =\sum_{ i = 0 }^ 10 \sum_{ j = 0 }^ (n + 1) i + j * x

function main() {

    var result_a1 = a1();
    var result_ax = a(5);
    var result_b = b(5, 3);
    var result_c = c(2, 3);
    var result_d = d(9);
    var result_e = e();
    var result_f = f();
    var result_g = g();
    var result_h = h(5);
    var result_k = k(5);
    var result_m = m(2, 5);  
}
