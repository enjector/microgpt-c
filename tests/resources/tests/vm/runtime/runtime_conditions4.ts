function main() {
    var amount = 100;

    if (amount < 100) {
        amount = amount - 10;
    }
    else if (amount >= 100) {
        amount = amount + 500;
    } else if (amount > 1000) {
        amount = 1000;
    }


    var amount2 = 2000;

    if (amount2 < 100) {
        amount2 = amount2 - 10;
    }
    else if (amount2 >= 100) {
        amount2 = amount2 + 500;
    } else if (amount2 > 1000) {
        amount2 = amount2 + 200;
    }
}