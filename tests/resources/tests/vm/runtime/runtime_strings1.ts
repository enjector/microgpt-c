function main(): string {
    var firstname = "Bob";
    var lastname = "Fish";
    var greeting = "Hello " + firstname + " " + lastname;

    var age = 18;
    var greeting2 = "You are " + age;

    //var greeting3 = age + "You are";

    var fullname = firstname;
    fullname = fullname + ", ";
    fullname = fullname + lastname;

    var greeting3 = "You are " + age + " is " + fullname;

    return "Reply " + greeting3;
}