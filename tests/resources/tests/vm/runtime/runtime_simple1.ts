function calculate_interest3(amount: number, rate: number): number {
    //var amount = 100;
    //amount = amount + 1.23;
    //var result = amount >= 101.23;
    return amount * rate;
}

function add_bonus(amount: number): number {
    return amount + 100;
}

function main(): number {
    var amount = 10;

    amount = 11;

    amount = amount + 120;

    // Do calculation
    var total = add_bonus(calculate_interest3(amount, 2));
    //debug(total);

    // Finished
    return total;
}