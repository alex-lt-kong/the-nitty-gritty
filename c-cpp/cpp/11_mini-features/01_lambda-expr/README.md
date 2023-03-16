# Lambda expression

* Introduced in C++11, Lambda expression (often called lambda) is just
anonymous function that is popular in JavaScript and Python.

* Sometimes, we still want to give a lambda expression a name, then we can call
it just like a normal function:

    ```C++
    auto func = []() {
        cout << "testOneNaiveLambda(): Hello, World!" << endl;
    };
    func();
    ```

    but we don't have to: we can just call it, without naming it:

    ```C++
    []() {
        cout << "testTwoAnonymousLambda(): Hello, World!" << endl;
    }();
    ```

* Lambda expression is also a typical case where the new `auto` keyword comes
in handy. In many cases, knowing the exact data type helps programmers to
think, but for lambda expressions, usually people don't care, or people are
well aware of that as the function's signature is just next to it.
    * There is one exception though: if we want to call the Lambda function
    recursively, we have to specify its type manually.

* Lambda expression is also the only way in C++ to create nested functions as
of C++17.

* Compared with JavaScript's anonymous function:
    ```JavaScript
    var func = () => {
        console.log("Hello, World!");
    };
    func();
    ```
    The C++'s version looks almost identical except the `[]` part.

* The `[]` part is called the "capture clause", the elements in the list within
`[]` are called captures. It is a way for the lamdba expression to access, or
"capture", variables from the surrounding scope.
    * Usage of the capture clause is detailed in `testFourCaptureClause()`
    [here](./main.cpp) 



## References

1. [Microsoft Learn: Lambda expressions in C++][1]

[1]: https://learn.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=msvc-170 "Microsoft Learn: Lambda expressions in C++"