/*
 * test.ts
 * Some useful info
 */

declare function debug(value: string): void;

function calculate_interest(amount: number, rate: number, max: number): number {
    var total = amount * rate;

    debug(max);

    if (!max) max = 100.1;

    if (total > max) {
        total = max;
    }

    debug(total);

    return total;
/*
 * test.ts
 * Some useful info
 */
}

function main(): number {
    var amount = 10.12;
/*
 * test.ts
 * Some useful info
 */

    // Do calculation
    var total = calculate_interest(amount, 1.12345);
    debug(total);

    total = calculate_interest(amount, 1.12345, 200.12);

    debug(total);

    // Finished

    return total;
}

/*
 * test.ts
 * Some useful info
 */
