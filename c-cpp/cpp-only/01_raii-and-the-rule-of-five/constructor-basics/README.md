# Constructor

## Copy constructor

- A copy constructor is a member function that initializes an object using
  another object of the same class. In simple terms, a copy constructor is used
  to create of "copy" of an existing object.

- Before talking about how a copy constructor is defined and used, we need to
  delve a bit deeper into the design of C++.

- One key mindset difference between C and C++ is that in C people mostly use
  pointers to refer to an "object", which has to be `malloc()`ed and `free()`ed
  manually. While in C++ objects behave like "common variables" such as integer,
  float, etc. For example:

  - Objects are passed by value by default, you pass an object to a function
    then modify the object, and the object being passed won't be changed.
  - Assigning an object to another object will create a new object, instead
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

  - Objects will go out of scope and are automatically released as long as
    it is defined on stack (i.e., `MyClass myObject = MyClass();`)
    instead of on heap (i.e., `MyClass myObject = new MyClass();`), just like
    normal variables (e.g., `int a = 5;`);
  - This design is also closely related to the concept of
    [RAII](../01_raii),
    which is discussed separately.

- This design also differs greatly from Java/C#, where almost all objects are
  reference types that behave like pointers except they are garbage collected.

- Copy constructor is an important function to support the above design--we
  need to frequently make copies of an object just like we make copies of an
  integer (e.g., `int a = b;`). There are a few cases where copy constructors
  will be called, including:

  - initialization: `T a = b;` or `T a(b);`, where b is of type T;
  - function argument passing: `f(a);`, where a is of type T and f is
    `void f(T t)`;
  - function return: `return a;` inside a function such as `T f()`, where a
    is of type T, which has no move constructor.

- If a class is not "too complicated" (what "too complicated" means is
  a more complex topic, let's skip it here), C++ compilers will implicitly create
  a copy constructor for us if we don't define it explicitly.
  - But we can always create a copy constructor ourselves, usually its
    signature is just `MyClass::MyClass(const MyClass& myObject);`
  - The implicit copy constructor could also be a trap. Say we have a
    raw pointer as a non-static member of a class, the implicit copy constructor
    could be added. But when we make a copy of the class, what will happen?
    A copy of the pointer, not the memory on the heap pointed by the pointer,
    i.e., a shallow copy, will be returned. If the original object owns the heap
    memory, if the original object goes out of scope, the copy object may
    refer to a memory address that is invalid.

## Move constructor and rvalue

### What is "rvalue" anyway?

- Before delving into the concept of move constructor, we may want to take a
  look at the dilemma move constructors try to solve.

- Even before we explore the dilemma, we need to have a rough idea of "rvalue".

  - For example, `x`, `y`, `z` below are **l**values as they are on the
    left-hand side of the statement:

  ```C
  int x = 0;
  int y = x * 2;
  int z = y;
  ```

  - while `x` and `y` below are still lvalue, `x+1`, `y*2` are **r**values:

  ```C
  int x = x + 1;
  int y = y * 2;
  ```

  - because you can't put them to the left of the assignment statement:

  ```C
  int x, y;
  x + 1 = 2; // WTF? This is non-sense
  y * 2 = 4;
  ```

  - Applying this principle, `5`, `"hello world!"`, etc. are also **r**value.

- One should be aware that the dichotomy or lvalue and rvalue is from the C
  world but it is a misnomer in the context in C++. Consider the following code:

```C++
std::string s = "Hello world!";
(s + s) = s;
```

(s + s) is an **r**value but it can legally appears on the LHS of a statement.
Perhaps it is more intuitive to call it just "temporary value" or "unnamed value"

### How can move constructor help?

- It is rather natural that the following should work:

  ```C++
  std::string s1 = "hello ";
  std::string s2 = "world!";
  // This seems fine, as s1 + s2 is a rvalue and
  // it rightfully appears on the rhs
  std::string foobar = s1 + s2;
  std::cout << foobar << std::endl;
  // >> hello world!
  ```

  - The above is nothing but syntactic sugar on top of this:

  ```C++
  std::string s1 = std::string("hello ");
  std::string s2 = std::string("world!");
  std::string foobar = std::string(s1 + s2);
  std::cout << foobar << std::endl;
  ```

- What does `s1` + `s2` return?

  - It returns a new anonymous `std::string` object that contains
    "hello world!". This unnamed new object is an rvalue.
  - Then, we will pass this rvalue to `std::string()`. Without a move
    constructor, the copy constructor of `std:string` will be called, which
    makes a copy of the rvalue and return the new object to `foobar` by value.

- But this is utterly wasteful--the anonymous `s1 + s2`, as an rvalue (i.e.,
  an unnamed temporary object), will be thrown away very soon, so why bother
  making a copy of it? Let's just transfer, a.k.a., "move", its content from the
  rvalue to `foobar`. Isn't it wonderful?

  - This is how a move constructor is used in
    [move-constructor.cpp](./move-constructor.cpp):

- There is a separate scenario where we want the move constructor to be called:

  ```C++
  std::string s1 = std::string("hello");
  // using std::move, we explicity transfer the ownership of "hello " from s1
  // to hell
  std::string hell = std::move(s1);
  cout << s1 << endl; // UB!
  cout << hell << endl; // prints "hello "
  ```

  - This is a bit similar to Rust.
  - Note that `std::move()` doesn't actually move anything out of it's own.
    It's just a fancy name for a cast to a `T&&`.

- Move constructors of all the types used with STL containers, for example,
  need to be declared `noexcept`; otherwise STL will choose copy constructors
  instead. The same is valid for move assignment operations.

### Why do we need rvalue reference for move constructor to work?

- A typical class with move constructor is like the below:

```C++
class MyClass {
private:
  ssize_t buf_size;
  void* buf_ptr;
public:
  MyClass(MyClass &&rhs) {
    if (this != &rhs) {
      buf_size = rhs.buf_size;
      buf_ptr = rhs.buf_ptr;
      rhs.buf_ptr = nullptr;
      rhs.buf_size = -1;
    }
    return *this;
  }
}
```

- The problem is, for move constructor, how about we just use **l**value
  reference, meaning that we change `MyClass(MyClass &&rhs)` to simply
  `MyClass(MyClass &rhs)`? Passing an object to it and it can still transfer
  the ownership.
  - For example:
  ```C++
  MyClass a = MyClass();
  // should work even if we change the move constructor's
  // signautre to from `MyClass(MyClass &&rhs)` to `MyClass(MyClass &rhs)`
  // as a is an lvalue.
  MyClass b = std::move(a);
  ```
- The superficial answer is that it may make it identical to copy constructor,
  which may cause difficulty for a compiler to pick the right method to call.
  But the real reason is more than this.
- The understand the reason, one needs to read
  [lvalue-vs-rvalue.cpp](./lvalue-vs-rvalue.cpp) and understand the design
  of C++. Long story short, **r**value reference makes it possible
  for us to modify an rvalue in the function call. This is needed by move
  semantics. Without introducing `T&&`, this is impossible: `const T&`
  accepts rvalue but it must be read-only and `T&` rejects rvalues altogether.

## Copy elision (a.k.a., return value optimization or RVO)

- Copy elision or RVO is a "minor" but very significant optimization.

- Consider the following case (Let's ignore function inline, compile-time
  computation, etc.):

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

  - What happens when we `return scores;`? By rights: `scores` is about to
    be out of scope and destroyed; a copy of `scores` is prepared and
    assigned to `myScores`.
  - This is worrying--what if `scores` is a vector with 1 million elements? We
    are going to copy all of them?

- After learning move constructor, we can make it a lot smarter by revising
  `GetScores()` to this:

  ```C++
  vector<double> GetScores() {
      vector<double> scores{ 1.414, 3.141, 2.71, 99, -1, 0.001 };
      return std::move(scores); // Let's move!
  }
  ```

  - But if we do try this and benchmark, we will see the performance dropping
    (compared with the navie version without `std::move()`), instead of going up.
    What happens?

- This is because ISO C++ standard has something beyond (and before
  the introduction of) the move constructor (and to a large extent makes move
  constructor much less common): copy elision.

  - It means that a C++ compiler can simply skip copy/move
    constructors altogether and just set the value directly to the object.
  - In the above case, it means something like this:

  ```C++
  int main() {
      vector<double> myScores = { 1.414, 3.141, 2.71, 99, -1, 0.001 };
      return 0;
  }
  ```

  - So there is no copy, no move, no nothing.

- Note that this is a violation of C++'s "as-if" rule.

  - The "as-if" rule requires that all optimization techniques must not
    change the observable behavior of a program.
  - One can try running `./move-constructor-noec.out` and
    `./move-constructor.out` (`-noec` means we enabled
    `-fno-elide-constructors`) and will notice that the output of the two
    programs is different.
  - This is not supposed to happen and it violates the "as-if" rule.

- Perhaps the C++ committee sees the benefit of copy elision as large enough
  to justify the exception so that a very explicit rule is added to the ISO C++
  standard to allow this counter-intuitive behavior.

- It is also worth noting that RVO is not guaranteed to be applied in all cases,
  for example:
  ```C++
  vector<double> GetScores() {
      vector<double> scores1{ 1.414, 3.141, 2.71, 99, -1, 0.001 };
      vector<double> scores2{ 1.414, 3.141, 2.71, 99, -1, 0.001 };
      vector<double> scores3{ 1.414, 3.141, 2.71, 99, -1, 0.001 };
      int a =
      if (rand() % 3 == 0) { return scores1; }
      else if (rand() % 3 == 1) { return scores2; }
      else { return scores3; }
  }
  ```
  - But compilers are still free to apply other optimization techniques such
    as function inlining.

## References

### Copy constructor

- [CPP Reference - Copy constructors](https://en.cppreference.com/w/cpp/language/copy_constructor)
- [Stackoverflow - Does C++ treat Class Objects like value types if initialized without the new operator?](https://stackoverflow.com/questions/13633824/does-c-treat-class-objects-like-value-types-if-initialized-without-the-new-ope)
- [GeeksForGeeks - Copy Constructor in C++](https://www.geeksforgeeks.org/copy-constructor-in-cpp/)

### Move constructor

- [Stackoverflow - What is move semantics?](https://stackoverflow.com/questions/3106110/what-is-move-semantics)
- [Microsoft Learn - Move Constructors and Move Assignment Operators (C++)](https://learn.microsoft.com/en-us/cpp/cpp/move-constructors-and-move-assignment-operators-cpp?view=msvc-170)
- [C++ Rvalue References Explained ](http://thbecker.net/articles/rvalue_references/section_01.html)
- [Stackoverflow - What is move semantics?](https://stackoverflow.com/questions/3106110/what-is-move-semantics)
- [Extra Clang Tools 17.0.0git documentation](https://clang.llvm.org/extra/clang-tidy/checks/performance/noexcept-move-constructor.html)

### Copy elision

- [Wikipedia - Copy elision](https://en.wikipedia.org/wiki/Copy_elision)
- [Working Draft, Standard for Programming Language C++](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/n4849.pdf)
- [Cppconference.com - Copy elision](https://en.cppreference.com/w/cpp/language/copy_elision)
- [stackoverflow.com - Move or Named Return Value Optimization (NRVO)?](https://stackoverflow.com/questions/6233879/move-or-named-return-value-optimization-nrvo)
