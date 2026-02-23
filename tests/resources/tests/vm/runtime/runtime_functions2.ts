// Simple functions

function test1() {
}

function test2() {
	return 1 + 2 * 3;
}

function test2(a) {
	return a;
}

function test3(a, b) {
	return a + b;
}

function test4(a, b) {
	return (a + b) + 2 * 9 + 100;
}

function main(): boolean {

	/**
	 * Run tests
	 */
	assert(test1() == undefined);
	assert(test2() == 7);
	assert(test3(3, 5) == 8);
	assert(test4(15, 19) == 152);

	return (test4(16, 19) == 152);
}