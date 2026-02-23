function calculate_interest3(amount: number, rate: number): number {
    return amount * rate;
}

function add_bonus(amount: number): number {
    return amount + 100;
}

function main() {
    var amount = 100;
    amount = amount + 1.23;


    var amount = 10;

    // Override test
    amount = 11;

    amount = amount + 120;

    // Do calculation
    var total = add_bonus(calculate_interest3(amount, 2.5));
    //debug(total);

    // Finished
}