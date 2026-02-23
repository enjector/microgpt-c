function test_a(amount: number): number {
    var reset = false;
    amount = amount + 1;

    if (amount < 100 || reset) {
        return 0;
    }

    return amount;
}

function test_b(amount: number): number {
    var check = amount;
    if (amount >= 100 || check > 9999) {
        return 0;
    } else {
        return 10;
    }
}

function main(): number {
    var amount = 100;

    test_a(amount);
    test_b(amount);

    return amount;
}