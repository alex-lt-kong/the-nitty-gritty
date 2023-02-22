# Value, Reference and Pointer

* It is a bit confusing for C++ to have both references and pointers.

* But before digging into the difference between references and pointers, we need to make one thing clear:
  * given C++'s object-oriented nature, it may be tempting to think that passing object to a function/method is done
  by passing its address/reference. It is wrong:
  ```C++
  class SimpleClass {
  private:
    int val;
  public:
  SimpleClass(int val) {
    this->val = val;  
  }
  ~SimpleClass() {}

  void setNewVal(int new_val) {
    this->val = new_val;
  }

  int getVal() {
    return this->val;
  }
  };

  void a_naive_function(SimpleClass simple_obj) {
    simple_obj.setNewVal(1234);
    printf("simple_obj.getVal()@a_naive_function(): %d\n", simple_obj.getVal());
  }

  void test_parameters_passing() {
    SimpleClass simple_obj = SimpleClass(10);
    printf("simple_obj.getVal()@test_parameters_passing(): %d\n", simple_obj.getVal());
    a_naive_function(simple_obj);
    printf("simple_obj.getVal()@test_parameters_passing(): %d\n", simple_obj.getVal());
  }
  ```
  * Will the third `printf()` show `10` or `1234`?
  ```
  simple_obj.getVal()@test_parameters_passing(): 10
  simple_obj.getVal()@a_naive_function(): 1234
  simple_obj.getVal()@test_parameters_passing(): 10
  ```
  * Simply put, unless we make the request explicitly, C++ passes arguments by value, not by address/pointer/reference,
  even if it may seem to be more "natural" for programs to pass objects as references/pointers.
    * This design also differs greatly from Java/C#, where almost all objects are reference types which behave like
    pointers except they are garbage collected.
  * But how is the object being able to be pass as value? It is simple: a copy of the object is created when it is 
  passed. This is done by a specific type of constructor called "copy constructor". Users can define a copy
  constructor explicitly, or the compiler will create one for us.
  * Using the above sample as an example, its copy constructor could be:
  ```C++
  SimpleClass(const SimpleClass& obj) {
    this->val = obj.val;
  }
  ```
  and running the code again outputs the following:
  ```
  simple_obj.getVal()@test_parameters_passing(): 10
  Copy constructor is called()
  simple_obj.getVal()@a_naive_function(): 1234
  simple_obj.getVal()@test_parameters_passing(): 10
  ```
  * From the example, one can observe that:
    * A copy constructor takes a reference (i.e., `SimpleClass& obj`) and it can't take a value. If it were
    allowed to take a value then what could happen? The copy constructor will need to call another copy constructor to
    make a copy, which doesn't make sense!
    * There aren't tricks such as [copy-on-write](#copy-on-write) (well at least at the source code level) and the copy constructor
    just naively copies everything.

* After making the above point clear, the difference between references and pointers are not that significant. As
a concrete example, the above function `a_naive_function()` can be revised to use either way:
  ```C++
  void a_naive_but_by_ref_function(SimpleClass& simple_obj) {
    simple_obj.setNewVal(1234);
    printf("simple_obj.getVal()@a_naive_but_by_ref_function(): %d\n", simple_obj.getVal());
  }

  void a_naive_but_by_pointer_function(SimpleClass* simple_obj) {
    simple_obj->setNewVal(1234);
    printf("simple_obj.getVal()@a_naive_but_by_pointer_function(): %d\n", simple_obj->getVal());
  }
  ```
  * Output:
    ```
    simple_obj.getVal()@test_pass_arguments_by_value(): 10
    simple_obj.getVal()@a_naive_but_by_ref_function(): 1234
    simple_obj.getVal()@test_pass_arguments_by_value(): 1234
    ...
    simple_obj.getVal()@test_pass_arguments_by_value(): 10
    simple_obj.getVal()@a_naive_but_by_pointer_function(): 1234
    simple_obj.getVal()@test_pass_arguments_by_value(): 1234
    ```
  * We are always free to convert a reference to a pointer:
    ```C++
    void a_naive_but_by_ref_with_ptr_function(SimpleClass& simple_obj) {
      (&simple_obj)->setNewVal(1234);
      printf("simple_obj.getVal()@a_naive_but_by_ref_function(): %d\n", (&simple_obj)->getVal());
    }
    ```
    ```
    simple_obj.getVal()@test_pass_arguments_by_value(): 10
    simple_obj.getVal()@a_naive_but_by_ref_with_ptr_function(): 1234
    simple_obj.getVal()@test_pass_arguments_by_value(): 1234
    ```
  * Note that in C++ `&`(ampersand/and sign) can be used in two ways: as the "address-of" operator and the reference declarator:
    ```C++
    int target;
    int &target_ref = target;  // target_ref is initialized to refer to target. That is, it is a reference to target.
    int* target_ptr = &target; // target_ptr is a pointer to target. That is, target_ptr stores the address of target.
    ````

* There are a few concrete differences between references and pointers, my take is a that reference is a less
flexible (i.e., less powerful) version of pointer. For example, a reference cannot:
  * be `NULL`/`nullptr`.
  * be re-assigned: A pointer can be pointed to different addresses many time while a reference cannot.
  * have arithmetic operations: `*(ptr + sizeof() * n)` is valid but there isn't an equivalent for references.
  (for sure we can do something like `*(&ref + sizeof() * n)`, but it means we fall back to pointers)


* There are arguments that "use reference wherever you can, pointers wherever you must."
  * People holding this view usually point out the downsides of pointers which I don't really agree--it is pointers
  that set C/C++ apart from other languages, without understanding/embracing the flexibility/peculiarity of
  pointers, why not using Java/C# instead?
  * My take on when to use reference and pointer when passing parameters is like this:
    * If we would like to make `NULL`/`nullptr` a valid possible value, use pointer; otherwise, if we want to treat
    an object like a primitive, use reference.
    * Also, there are a few cases where using references is mandatory, such as copy constructors and operator
  overloading.

## Copy-on-write

* Copy-on-write (CoW) is a loosely related topic which can also be covered here.
  * CoW is mostly an implmentation details. As users of C++ we don't usually need to implement it. Being
  aware of its existence and implication shall mostly suffice.
  * According to [this answer](https://stackoverflow.com/questions/1649028/how-to-implement-copy-on-write), in a
  multi-threaded environemnt (which is most of them nowadays) CoW is frequently a huge performance hit rather
  than a gain. And with careful use of const references, it's not much of a performance
  gain even in a single threaded environment.
  * CoW is tricky to implement, and it's easy to make mistakes. CoW is also responsible for
  [Dirty CoW](https://en.wikipedia.org/wiki/Dirty_COW), named as the
  ["Most serious" Linux privilege-escalation bug](https://arstechnica.com/information-technology/2016/10/most-serious-linux-privilege-escalation-bug-ever-is-under-active-exploit/)
  by some industry media.

* The idea of CoW is this:
  ```C++
  std::string x("Hello");
  std::string y = x;  // x and y use the same buffer.
  y += ", World!";    // Now y uses a different buffer; x still uses the same old buffer.
  ```
  * Note that CoW implementation of `std::string` is "allowed" (not mandatory) in C++98 and banned in C++11.
  * `g++`'s C++98 version does not implement string with CoW.
    ```C++
    void test_cow_in_stl() {
      string str1("Hello!");
      string str2 = str1;
      printf("%s, %s\n", str1.c_str(), str2.c_str());
      printf("%p, %p\n", str1.data(), str2.data());
    }
    ```
    ```
    Hello!, Hello!
    0x7ffd53de7450, 0x7ffd53de7470
    ```