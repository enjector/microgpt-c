function main() {
    var amount = 100;

    var a = 0;
    var b = 0;
    var c = 0;
    var d = 0;
    var e = 0;

    var a1 = 0;
    var b1 = 0;

    var f = 0;
    var g = 0;

    if (amount > 10) {
        a = 1;

        if (amount >= 100) {
            b = 1;
        }
        else {
            b1 = 1;
        }
    }
    else {
        a1 = 1;
    }

    if (amount - 1 == 99) {
        c = 1;
    }

    if (amount < 200) {
        if (amount <= 100) {
            d = 1;
        }
    }

    if (amount > 100) {
        e = 1;
    }

    if (amount < 1000) {
        f = 1;
        f = f + 1;
    }
    else {
        g = 1;
    }
}