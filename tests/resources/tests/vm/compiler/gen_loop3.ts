function main() {
    var total = 100;

    var finished = false;
    var i = 10;

    while ((i < 100 && total < 100) || !finished) {
        i = i + 10;
        total = total + 10;

        if (i == total) {
            finished = true;
        }
    }
}