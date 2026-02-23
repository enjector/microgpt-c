function calculate_interest3(amount: number, rate: number): number {
    return amount * rate;
}

function add_bonus(amount: number): number {
    return amount + 100;
}

function main() {
    var amount = 10;

    // test
    amount = amount + 120;

    // Do calculation
    var total = add_bonus(calculate_interest3(amount, 1.12345));
    //debug(total);

    // Finished
}