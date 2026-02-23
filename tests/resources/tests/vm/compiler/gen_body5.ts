// Test test.js

/*
 * test.ts
 * Some useful info
 */


// Comment 1a

// Comment 1b
function wrong_return_type(amount: number): boolean {
    return 99.133;
}

// Comment 1a
// Comment 1b
function wrong_return_type2(): number {
/* This is a test
that wraps
multiple lines */
    var var1 = true;

    return var1;
}

function my_function(): number {
    return 123;
}

function main() {
    var amount = 10.12;

    // Wrong assignment
    amount = wrong_return_type(1234.1);

    var z = true;

    z = my_function();
}

// Comment 1b

/*
 * test.ts
 * Some useful info
 */
