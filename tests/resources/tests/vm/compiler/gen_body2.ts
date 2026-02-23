declare function debug(value: number) : void;

function get_rate(country: string): number {
    return 1.34;
}

function calculate_interest2(amount: number, country: string): number {
    return amount * get_rate() * global_rate;
}

function dosomething(name: string) {
    var a = 1;
}

function main() {
    var amount = 100;

    amount = amount + 100;

    var interest = calculate_interest2(amount, "UK");
    debug(interest);

    interest = calculate_interest2("USA", amount);
    debug(interest);
}