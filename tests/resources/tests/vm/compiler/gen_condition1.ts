function main(): boolean {
    var amount = 100;

    if (amount < 50) amount = 0;

    if (amount < 100) {
        amount = 0;
    }

    if (amount >= 100) {
        amount = 0;
    }

    var completed = amount != 0;

    return completed;
}