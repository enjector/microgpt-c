function main(): number {

    var FullName = "Fred Smith";

    var a = 0;
    var b = 0;
    var c = 0;
    var d = 0;
    var e = 0;
    var f = 0;
    var g = 0;
    var h = 0;
    var i = 0;
    var j = 0;

    if (FullName == "Fred Smith") {
        a = 1;
    }
    else {
        b = 1;
    }

    if (FullName != "Fred Smith") {
        c = 1;
    }
    else {
        d = 1;
    }

    if (FullName != "Smith") {
        e = 1;
        f = 1;
    }
    else {
        g = 1;
        h = 1;
    }

    if (a == 1 && b == 1) {
        i = 1;
    }

    if (a == 1 || b == 1) {
        j = 1;
    }

    return j;
}
