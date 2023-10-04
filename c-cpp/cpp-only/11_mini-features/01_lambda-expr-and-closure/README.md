# Lambda expression and closure

* Introduced in C++11, Lambda expression (often called lambda) is just
anonymous functions that are popular in JavaScript and Python:

    ```JavaScript
    var func = () => {
        console.log("Hello, World!");
    };
    func();
    ```

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

* Compared with JavaScript's anonymous function, the C++'s version looks
almost identical except the `[]` part.

* The `[]` part is called the "capture clause", the elements in the list within
`[]` are called captures. It is a way for the lamdba expression to access, or
"capture", variables from the surrounding scope.
    * Usage of the capture clause is detailed in `testFourCaptureClause()`
    [here](./main.cpp) 


## Closure

* A lot of definitions of closure is rather confusing. Take the introduction
on Wikipedia as an example:

    > In programming languages, a closure, also lexical closure or function
    > closure, is a technique for implementing lexically scoped name binding
    > in a language with first-class functions. Operationally, a closure
    > is a record storing a function[a] together with an environment.

    * WTF does this nonsense mean???

* In JavaScript, this is bloody simple:

    ```JavaScript
    function makeAdder(x) {
        return (y)=>{
            return x + y;
        };
    }

    const add5 = makeAdder(5);
    const add10 = makeAdder(10);

    console.log(add5(2)); // 7
    console.log(add10(2)); // 12
    ```

    * When we get a "copy" of the anonymous function nested in `makeAdder(x)`,
    it "remembers" the value being passed to its parent function and it just
    works.
    * In JavaScript's term, we say:
        > `add5` and `add10` both form closures. They share the same function body
        > definition, but store different lexical environments. In `add5`'s
        > lexical environment, `x` is 5, while in the lexical environment for
        > `add10`, `x` is 10.

* We replicate the same in C++ in `testSixClosure()`  [here](./main.cpp).
    * The thing becomes a bit fuzzy now. My understanding is that `func0`,
    `func1` and `func2` are all closures as closures are "unnamed function
    objects capable of capturing variables in scope."
    * This differs from the opinion of [Lei Mao][2].
    * But one way or another, C++'s design could result in undefined behavior
    if variables are captured by reference by a lambda expression and the
    captured-by-reference variable is accessed after its lifetime ended (shown
    in `func2()@testSixClosure()`).[3][3]

* In C++, Lambdas are just syntactic sugar for instances of classes that
overload "function call operator" (`operator()()`)[4][4].
    * This helps us to understand the nature of "closure" as well--think
    captures as parameters passed to the constructor of the instance then the
    closure is nothing but an instance to a class--it contains some variables
    of course and even if its caller goes out of scope the instance still keeps
    its copy of data, forming a "closure".

## References

1. [Microsoft Learn: Lambda expressions in C++][1]
2. [C++ Closure][2]
3. [Lambda expressions][3]
4. [Lambdas are just classes with operator() overloaded][4]
5. [C++ Lambda Under the Hood][5]

[1]: https://learn.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=msvc-170 "Microsoft Learn: Lambda expressions in C++"
[2]: https://leimao.github.io/blog/CPP-Closure/ "C++ Closure"
[3]: https://en.cppreference.com/w/cpp/language/lambda "Lambda expressions"
[4]: https://stackoverflow.com/questions/45831371/lambdas-are-just-classes-with-operator-overloaded "Lambdas are just classes with operator() overloaded?"
[5]: https://medium.com/software-design/c-lambda-under-the-hood-9b5cd06e550a "C++ Lambda Under the Hood"