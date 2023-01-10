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
    [RAII](https://github.com/alex-lt-kong/the-nitty-gritty/tree/main/cpp/01_raii),
    which is discussed separately.

* This design also differ greatly from Java/C#, where almost all objects are
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


## References

* [CPP Reference - Copy constructors](https://en.cppreference.com/w/cpp/language/copy_constructor)
* [Stackoverflow - Does C++ treat Class Objects like value types if initialized without the new operator?](https://stackoverflow.com/questions/13633824/does-c-treat-class-objects-like-value-types-if-initialized-without-the-new-ope)
* [GeeksForGeeks - Copy Constructor in C++](https://www.geeksforgeeks.org/copy-constructor-in-cpp/)