# Constructor

## Copy constructor

* A copy constructor is a member function that initializes an object using
another object of the same class. In simple terms, a copy constructor is used
to create of "copy" of an existing object.

* Before talking about how copy constructor is defined and used, we need to
delve in a bit deeper on the design of C++.

* One key mindset difference between C and C++ is that in C people mostly use
pointers to refer to an "object", which has to be `malloc()`ed and `free()`ed
manually. While in C++ objects behave like "common variables" such as integer,
float, etc. For example:
    * Objects are passed by value by default, you pass an object to a function
    then modify the object, the object being passed won't be changed.
    * Assigning an object to another object will create a new object, instead
    of creating a pointer pointing to the same memory location.

    ```C++
    MyItemType a;
    MyItemType b;
    a.someNumber = 5;
    b = a;

    cout << a.someNumber << endl;
    cout << b.someNumber << endl;

    b.someNumber = 10;

    cout << a.someNumber << endl;
    cout << b.someNumber << endl;
    ``` 
    Output:
    ```
    5  
    5
    5
    10
    ```
    * Objects will go out of scope and being automatically released as long as
    it is defined on stack (i.e., `MyClass myObject = MyClass();`)
    instead of on heap (i.e., `MyClass myObject = new MyClass();`), just like
    normal variables (e.g., `int a = 5;`);
    * This design is also closely related to the concept of
    [RAII](../01_raii),
    which is discussed separately.

* This design also differs greatly from Java/C#, where almost all objects are
reference types which behave like pointers except they are garbage collected.

* Copy constructor is an important function to support the above design--we
need to frequently make copies of an object just like we make copies of integer
(e.g., `int a = b;`). There are a few cases where copy constructors will be
called, including:
    * initialization: `T a = b;` or `T a(b);`, where b is of type T;
    * function argument passing: `f(a);`, where a is of type T and f is
    `void f(T t)`;
    * function return: `return a;` inside a function such as `T f()`, where a
    is of type T, which has no move constructor.  


* If a class is not "too complicated" (what does "too complicated" means is 
a difficult topic, let's skip it here), C++ compilers will implicitly create
a copy constructor for us, if we don't define it explicitly.
    * But we can always create a copy constructor ourselves, usually its
    signature is just `MyClass::MyClass(const MyClass& myObject);`
    * The implicit copy constructor could also be a trap. Say we have a
    raw pointer as a non-static member of a class, the implicit copy constructor
    could be added. But when we make a copy of the class, what will happen?
    A copy of the pointer, not the memory on heap pointed by the pointer, i.e.,
    a shallow copy, will be returned. If the original object owns the heap
    memory, if the original object goes out of scope, the copy object may
    refer to a memory address that is invalid.


## Move constructor

* Before delving into the concept of move constructor, we may want to take a
look at the dilemma move constructors tries to solve.

* Even before we explore the dilemma, we need to have a rough idea of "rvalue".
    * For example, `x`, `y`, `z` below are **l**values as they are on the
    left-hand side of the statement:
    ```C
    int x = 0;
    int y = x * 2;
    int z = y;
    ```
    while `x` and `y` below are still lvalue, `x+1`, `y*2` are **r**values:
    ```C
    int x = x + 1;
    int y = y * 2;
    ```
    because you can't put them to the left of the assignment statement:
    ```C
    int x, y;
    x + 1 = 2; // WTF? This is non-sense
    y * 2 = 4;
    ```

* While `x + 1 = 2` is non-sense, we may agree that the following should work:

    ```C++
    std::string hel = "hello ";
    std::string wor = "world!";
    // This seems fine, as hel + wor is a rvalue and 
    // it rightfully appears on the rhs
    std::string foobar = hel + wor; 
    std::cout << foobar << std::endl;
    // >> hello world!
    ```

    * The above is nothing but syntactic sugar on top of this:

    ```C++
    std::string hel = std::string("hello ");
    std::string wor = std::string("world!");
    std::string foobar = std::string(hel + wor);
    std::cout << foobar << std::endl;
    ```


* What does `hel` + `wor` return?
    * It returns a new anonymous `std::string` object that contains
    "hello world!". This unnamed new object is a rvalue.
    * Then, we will pass this rvalue to `std::string()`, invoking the copy
    constructor of `std:string`, which will make a copy of the rvalue and
    return the new object to `foobar` by value.
    * Side note: `str::string` must overload the `+` operator to make
    `hel + wor` work.

* But this is utterly wasteful--the anonymous `hel + wor` will be thrown away
very soon, why bother making a copy of it? Let's just transfer, a.k.a., "move",
its content from the rvalue to `foobar`. Isn't it wonderful?
    * This is how a move constructor is used in [move-constructor.cpp](./move-constructor.cpp):

    ```C++    
    NaiveString(NaiveString&& rhs) {
        _str = rhs._str; // "Ownership transfer"
        // Note that we need to set the rhs's internal pointer to nullptr,
        // so that when the rvalue goes out of scope, it won't release my
        // resource (it's ownership has been transferred)
        rhs._str = nullptr;
    }
    ```

    * `NavieString&& rhs` is called an `rvalue reference` introduced in C++11.

* There is a separate scenario where we want the move constructor to be called:

    ```C++
    std::string hel = std::string("hello");
    // using std::move, we explicity transfer the ownership of "hello " from hel
    // to hell
    std::string hell = std::move(hel);
    cout << hel << endl; // UB!
    cout << hell << endl; // prints "hello "
    ```

    * This is a bit similar to Rust.


## Copy elision (a.k.a., return value optimization or RVO)

* Copy elision or RVO is a "minor" but very significant optimization.

* Consider the following case (Let's ignore function inline, compile-time
computation, etc):
    ```C++
    vector<double> GetScores() {
        vector<double> scores{ 1.414, 3.141, 2.71, 99, -1, 0.001 };
        return scores;
    }
    int main() {
        vector<double> myScores = GetScores();
        return 0;
    }    
    ```
    * What happens when we `return scores;`? By rights: `scores` is about to
    be out of scope and destroyed; a copy of `scores` is prepared and 
    assigned to `myScores`.
    * This is worrying--what if `scores` is a vector with 1 million elements? We
    are going to copy all of them?

* After learning move constructor, we can make it a lot smarter by revising
    `GetScores()` to this:
    ```C++
    vector<double> GetScores() {
        vector<double> scores{ 1.414, 3.141, 2.71, 99, -1, 0.001 };
        return std::move(scores); // Let's move!
    }
    ```
    * But if we do try this and benchmark, we will see the performance dropping,
    instead of going up. What happens?

* This is because ISO C++ standard has something beyond move constructor (and
to a large extent makes move constructor much less common): copy elision.
    * It means that a C++ compiler can simply skip copy/move
    constructors altogether and just set the value directly to the object.
    * In the above case, it means something like this:

    ```C++
    int main() {
        vector<double> myScores = { 1.414, 3.141, 2.71, 99, -1, 0.001 };
        return 0;
    }    
    ```
    * So there is no copy, move, no nothing.

* Note that this is a violation of C++'s "as-if" rule.
    * The "as-if" rule requires that all optimization techniques must not
    change the observable behavior of a program.
    * One can try running `./move-constructor-noec.out` and
    `./move-constructor.out` (`-noec` means we enabled
    `-fno-elide-constructors`) and will notice that the output of the two
    programs is different.
    * This is not supposed to happen and it violates the "as-if" rule.

* Perhaps the C++ committee sees the benefit of copy elision as large enough
to justify the exception so that a very explicit rule is added to the ISO C++
standard to allow this counter-intuitive behavior.


## References

### Copy constructor
* [CPP Reference - Copy constructors](https://en.cppreference.com/w/cpp/language/copy_constructor)
* [Stackoverflow - Does C++ treat Class Objects like value types if initialized without the new operator?](https://stackoverflow.com/questions/13633824/does-c-treat-class-objects-like-value-types-if-initialized-without-the-new-ope)
* [GeeksForGeeks - Copy Constructor in C++](https://www.geeksforgeeks.org/copy-constructor-in-cpp/)

### Move constructor
* [Stackoverflow - What is move semantics?](https://stackoverflow.com/questions/3106110/what-is-move-semantics)
* [Microsoft Learn - Move Constructors and Move Assignment Operators (C++)](https://learn.microsoft.com/en-us/cpp/cpp/move-constructors-and-move-assignment-operators-cpp?view=msvc-170)
* [C++ Rvalue References Explained ](http://thbecker.net/articles/rvalue_references/section_01.html)
* [Stackoverflow - What is move semantics?](https://stackoverflow.com/questions/3106110/what-is-move-semantics)

### Copy elision
* [Wikipedia - Copy elision](https://en.wikipedia.org/wiki/Copy_elision)
* [Working Draft, Standard for Programming Language C++](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/n4849.pdf)
* [Cppconference.com - Copy elision](https://en.cppreference.com/w/cpp/language/copy_elision)
* [stackoverflow.com - Move or Named Return Value Optimization (NRVO)?](https://stackoverflow.com/questions/6233879/move-or-named-return-value-optimization-nrvo)