function eval(loan_amount: number, interest_rate: number, duration_months: number): number {

    var apr = (loan_amount * interest_rate * (1 + interest_rate) * duration_months) / ((1 + interest_rate) * duration_months - 1);

    return apr;
}
