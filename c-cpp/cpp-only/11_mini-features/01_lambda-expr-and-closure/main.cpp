#include <iostream>
#include <functional>

using namespace std;

void testOneNaiveLambda() {
    /* Unfortunately, this isn't really anonymous--we gave the function
    a name: func*/
    auto func = []() {
        cout << "testOneNaiveLambda(): Hello, World!" << endl;
    };
    func();
}

void testTwoAnonymousLambda() {
    /* This is anonymous, because we didn't give the function a name */
    []() {
        cout << "testTwoAnonymousLambda(): Hello, World!" << endl;
    }();
}

void testThreeLambdaWithArgs() {
    auto func = [](int x, int y) {
        // Lambda expr can have arguments, just like a normal function
        cout << "testThreeLambdaWithArgs(): x = " << x << endl;
        cout << "testThreeLambdaWithArgs(): y = " << y << endl;
    };
    func(1, 2);
}

void testFourCaptureClause() {
    int x = 0;
    int y = 0;
    [&x, &y]() {
        /* Capture by reference, we can modify the value in the lambda.
           Each item in the "captures" list is called a "capture". */
        cout << "testFourCaptureClause(): x = " << x << endl;
        cout << "testFourCaptureClause(): y = " << y << endl;
        ++x;
        ++y;
    }();
    [&]() {
        /* Capture all varaibles by reference, this is called "capture-default"
        */
        cout << "testFourCaptureClause(): x = " << x << endl;
        cout << "testFourCaptureClause(): y = " << y << endl;
        ++x;
        ++y;
    }();
    [=]() {
        /* & is one "capture-default", = is another "capture-default", it means
           to capture all varaibles by value, meaning that we can access,
           but not modify. */
        cout << "testFourCaptureClause(): x = " << x << endl;
        cout << "testFourCaptureClause(): y = " << y << endl;
        /*
        ++x;
        ++y;
        */
    }();
    
    [=, &y]() {
        /* Capture all varaibles by value, except y by reference.  */
        ++y;
        cout << "testFourCaptureClause(): x = " << x << endl;
        cout << "testFourCaptureClause(): y = " << y << endl;
    }();
    []() { // Not capture any variable, we can't access them at all.
        /*
        cout << "testFourCaptureClause(): x = " << x << endl;
        cout << "testFourCaptureClause(): y = " << y << endl;        
        ++x;
        ++y;
        */
    }();
    
}

void testFiveRecursiveLambda() {
    function<size_t(size_t)> factorial; // No, we cant use auto here.

    factorial = [&](size_t a) -> int {
        if (a == 1) {
            return 1;
        }
        else {
            return a * factorial(a - 1);
        }
    };
    cout << "testFiveRecursiveLambda():" << endl;
    cout << "factorial(8): " << factorial(8) << endl;
}

auto closureWrapper1()
{
    int x = 10;
    return [x](){ std::cout << "Value in the closure: " << x << std::endl;};
}

function<void(void)> closureWrapper2()
{
    int x = 10;
    return [&x](){x += 1; std::cout << "Value in the closure: " << x << std::endl;};
}

auto testSixClosure() {
    int x = 10;
    auto func0 =  [&x](){x += 1; std::cout << "Value in the closure: " << x << std::endl;};
    cout << "testSixClosure():" << endl;
    auto func1 = closureWrapper1();
    auto func2 = closureWrapper2();
    func0();
    func0();
    func1();
    func1();
    // UB!!! Capture x by reference and it goes out of scope when
    // func2() is called!
    func2();
    func2();
}

int main() {
    testOneNaiveLambda();
    cout << endl;
    testTwoAnonymousLambda();
    cout << endl;
    testThreeLambdaWithArgs();
    cout << endl;
    testFourCaptureClause();
    cout << endl;
    testFiveRecursiveLambda();
    testSixClosure();
    return 0;
}