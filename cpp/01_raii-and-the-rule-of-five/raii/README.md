# Resource acquisition is initialization

- RAII is an awkward name for a useful feature. One can understand it as C++'s
  answer to a `finally` keyword in Python, etc.

- Let's consider this Python snippet:

  ```Python
  try:
      conn = pyodbc.connect(conn_str)
      cursor = conn.cursor()
      # db operations
  except Exception as ex:
      print(ex)
  finally:
      if conn is not None:
          conn.close()
  ```

    - `finally` is needed because we don't want `conn` to be left open whether
      or not an exception is thrown.

- In comparison, RAII is rather straightforward, let's wrap the resource
  releasing
  in destructor and always call destructor even if an exception is thrown:

  ```C++
  class DatabaseHelper {
  public:
  DatabaseHelper(RawDBConn* rawConn_) : rawConn(rawConn_) {};
  ~DatabaseHelper() {delete rawConn; }
  ... // omitted operator*, etc
  private:
  RawDBConn* rawConn;
  };

  DatabaseHelper handle(createNewResource());
  handle->performInvalidOperation();
  ```

    - The "official" opinion from
      [Bjarne Stroustrup](https://www.stroustrup.com/bs_faq2.html#finally)
      also argues that RAII is almost always better than a "finally" keyword.

    - Theoretically, RAII means "to bind the life cycle of a
      resource that must be acquired before use (allocated heap memory, thread
      of execution, open socket, open file, locked mutex, disk space,
      database connection—anything that exists in limited supply) to the
      lifetime
      of an object.".
        - [This Microsoft tutorial](https://learn.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=msvc-170)
          summarizes the RAII principle as "to give ownership of any
          heap-allocated resource—for example, dynamically-allocated memory or
          system object handles—to a stack-allocated object whose destructor
          contains the code to delete or free the resource and also any
          associated cleanup code."
    - To be specific, RAII encapsulates each resource into a class, where:
        - the constructor acquires the resource and establishes all class
          invariants or throws an exception if that cannot be done;
        - the destructor releases the resource and never throws exceptions;

- What if constructors or destructors throw exceptions? Things get a bit
  messier here.

    - In case of exceptions thrown by a constructor, the object’s destructor is
      _not_ invoked. If your object has already done something that needs to be
      undone (such as allocating some memory, opening a file, or locking a
      semaphore), this "stuff that needs to be undone" must be manually released
      in the constructor's error handling code.
    - But why the destructor is not called if an exception is thrown within
      a constructor? In C++, the lifecycle of an object starts when its
      initialization is done (roughly the same as "constructor returns
      successfully"). If an exception is raised in its constructor, the object's
      lifecycle has not started yet, so its destructor will not be called.
    - In case of destructors--
      [destructors should never fail](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Re-never-fail).
      If some function calls in a destructor could possibly throw exceptions, we
      should handle the exception within destructor, rather than propagating
      the exception up the call stack.

- Apart from being handy for database connection management, etc
  RAII is also the underlying principle
  of [smart pointers](../10_smart-pointers).
  Say we use a class to wrap a raw pointer, using constructor to `malloc()`
  memory and use its destructor to `free()` the memory--brilliant! a
  "smart pointer" already!

## Pointers with RAII and [the rule of three](https://en.cppreference.com/w/cpp/language/rule_of_three)

- Before exploring this section, you may want to take a look at the examples
  under the [constructor-basics](./constructor-basics/) directory.

- So far, so good--playing in the C++ playground, things are nice and tidy.
  However, as always, it becomes more interesting (or horrible if you prefer)
  when
  C comes into play.

- What if, for whatever reason, we need to `malloc()` two, instead of one, raw
  pointers in a constructor and `free()` them in destructors? Going down
  this rabbit hole, things start to turn surreal.

- Think about this naive implementation. Memory is `malloc()`ed in constructor
  and `free()`ed in destructor, perfect RAII:

  ```C++
  class Wallet {
  private:
    int *ptr0 = nullptr;
    int *ptr1 = nullptr;
    size_t wallet_size = 0;
    void mallocPtrs() {
      ptr0 = (int*)malloc(sizeof(int) * wallet_size);
      ptr1 = (int*)malloc(sizeof(int) * wallet_size);
    }
  public:
    Wallet (size_t wallet_size) {
      this->wallet_size = wallet_size;
      mallocPtrs();
    }
    ~Wallet () {
      free(ptr0);
      free(ptr1);
    }

  };
  ```

    - The code should work 99.9% of time if both `malloc()`s do not fail. But
      what if _both_ of them fail? It means `ptr0` and `ptr1` will be
      both `NULL`, and any subsequent dereference could cause unexpected
      results! We need to prevent this.

- A slightly better version. Now we throw `bad_alloc` exception if `malloc()`
  fails. This should prevent NULL pointers dereference:

      ```C++
      class Wallet {
      private:
        int* ptr0;
        int* ptr1;
        void mallocPtrs() {
          ptr0 = (int*)malloc(sizeof(int) * wallet_size);
          if (ptr0 == NULL) { throw std::bad_alloc(); }
          ptr1 = (int*)malloc(sizeof(int) * wallet_size);
          if (ptr1== NULL) { throw std::bad_alloc(); }
        }
      public:
        Wallet (size_t wallet_size) {
          this->wallet_size = wallet_size;
          mallocPtrs();
        }
        ~Wallet () { /**/ }
      };
      ```

    - Well yes, it doesn't suffer from NULL pointer dereference. Awesome!
      What if the first `malloc()` succeeds and the second `malloc()`
      fails? A `bad_alloc` exception will be thrown, RAII's principle is
      followed.
      But wait, what happens to the large amount of memory `malloc()`ed for and
      pointed by `ptr0`? No one takes care of it. It will be left on the heap
      lonely, forever! No, we need to handle this as well.

- Another version is prepared to handle this. Now we will manually `free()`
  the memory `malloc()`ed for the 1st pointer in case only the 2nd `malloc()`
  fails:

  ```C++
  class Wallet {
  private:
    int* ptr0 = nullptr;
    int* ptr1 = nullptr;
    void mallocPtrs() {
      ptr0 = (int*)malloc(sizeof(int) * wallet_size);
      if (ptr0 == NULL) { throw std::bad_alloc(); }
      ptr1 = (int*)malloc(sizeof(int) * wallet_size);
      if (ptr1== NULL) {
        // handle the already malloc()'ed pointer manually.
        free(ptr0);
        throw std::bad_alloc();
      }
    }
  public:
    Wallet (size_t wallet_size) {
      this->wallet_size = wallet_size;
      mallocPtrs();
    }
    ~Wallet () { /**/ }
  };
  ```

### Copy constructor

- We are done? Not yet! Think about this common usage:

  ```C++
  Wallet first_wallet = Wallet();
  first_wallet.ptr0[0] = 1;
  first_wallet.ptr1[2] = 2147483647;
  // We want "fill in" second_wallet by making a copy of first_wallet
  Wallet second_wallet = first_wallet;
  // Changing the data in second_wallet should not have any impact on the
  // data in first_wallet, for sure.
  second_wallet.ptr0[0] = 31415926;
  second_wallet.ptr1[2] = -1;
  ```

    - As we don't define a copy constructor, the compiler will prepare one
      for us. However, the compiler doesn't know if we want to copy `ptr0` and
      `ptr1` by reference or by value. But as we have two integer pointers,
      the default copy constructor will copy only the value of these pointers.
      That is, it is copy by reference.
      The most significant implication is that two seemingly distinct objects
      will share the same buffers. Meaning that changing one object's value
      will impact the other's.
    - But it is more than this, running the above code results in something
      like the below:

  ```C++
  first_wallet: 1, 2147483647
  first_wallet: 31415926, -1
  second_wallet: 31415926, -1
  Segmentation fault
  ```

    - Well, we accept that there will be buffer sharing, but WTH is there a
      Segfault?? It has to do with C++'s automatic memory management. When an
      object goes out of scope, its destructor is automatically called. In the
      above sample, `first_wallet` and `second_wallet` go out of scope one
      after another and their destructor is called. This is what RAII requires
      as well.
    - After `first_wallet` goes out of scope, we rightfully `free()` both
      pointers. But wait, what happens when `second_wallet`'s destructor is
      called? `ptr0` and `ptr1` are being `free()`ed again, it's
      a [double free](https://encyclopedia.kaspersky.com/glossary/double-free/)!
      Nonono, this shouldn't happen. We need to prepare a
      [copy constructor](../rule-of-five/constructors.cpp) for it:

  ```C++
  class Wallet {
  private:
    /**/
    void mallocPtrs() { /**/ }
  public:
    /* ... */
    // Copy constructor
    Wallet(const Wallet& rhs) {
      mallocPtrs();
      memcpy(ptr0, rhs.ptr0, sizeof(int) * wallet_size);
      memcpy(ptr1, rhs.ptr1, sizeof(int) * wallet_size);
    }
    /* ... */
  };
  ```

### Copy assignment operator and the rule of three

- If `second_wallet` is not initialized and we want to fill data into it from
  an existing `Wallet` object, we need a copy constructor, but if
  `second_wallet`
  has already been initialized and we want to replace its data with the data
  from another `Wallet` object (and properly handle memory resources of course),
  copy constructor won't help--as the object has already been initialized, its
  constructor won't be called again. In this case, we need a copy assignment
  operator.

- Together with a copy constructor and a destructor, it is known as
  [the rule of three](https://en.cppreference.com/w/cpp/language/rule_of_three):

  ```C++
  class Wallet {
  private:
    /**/
    void mallocPtrs() { /**/ }
  public:
    /* ... */
    // Copy constructor
    Wallet(const Wallet& rhs) {
      mallocPtrs();
      memcpy(ptr0, rhs.ptr0, sizeof(int) * wallet_size);
      memcpy(ptr1, rhs.ptr1, sizeof(int) * wallet_size);
    }
    // Copy assignment operator
    Wallet &operator=(const Wallet &rhs) {
      if (this != &rhs) {                     // not a self-assignment
        if (wallet_size != rhs.wallet_size) { // resource cannot be reused
          this->~Wallet();
          wallet_size = rhs.wallet_size;
          mallocPtrs();
        }
        memcpy(ptr0, rhs.ptr0, sizeof(int) * wallet_size);
        memcpy(ptr1, rhs.ptr1, sizeof(int) * wallet_size);
      }
      return *this;
    }
    /* ... */
  };
  ```

- A copy assignment operator is more complicated than a copy constructor
  because we need to manually release the resources already been allocated to
  the object.

- Another significant feature is the "self-assignment guard" that
  handles the below use case:

  ```C++
  Wallet first_dwallet = Wallet(2048);
  first_dwallet.ptr0[0] = 3;
  first_dwallet.ptr1[2047] = 666;
  first_dwallet = first_dwallet // self-assignment!
  ```

    - Without self-assignment check, the memory will be wiped out and we will
      copy from nowhere.

### Exception safety

- Using the copy assignment operator as an example, let's think about what
  could happen if an exception is thrown in `mallocPtrs()`: as `ptr0` and `ptr1`
  have both been `free()`ed, and `mallocPtrs()` `free()`s `ptr0` in case of
  `malloc()` fails to allocate memory to `ptr1`, there shouldn't be any memory
  leak.

    - In this scenario, we say the implementation provides basic exception
      guarantee.
    - However, `*this` object is left in an unusable state and its previous
      data have already been wiped out--this is not something we want. Ideally,
      if the system fails to allocate resources for new data, old data should be
      kept intact--just like the "transaction" concept in SQL database.
    - If we are able to keep the old data intact in case of exception being
      thrown in copy assignment operator, we say the function provides strong
      exception guarantee.
    - Current implementation does not provide strong exception guarantee but it
      does provide basic exception guarantee.

- Providing strong exception guarantee comes at a cost--we need to create a
  temporary object and then swap `*this`'s resources with the temporary object's
  resources. This is not always easy to implement and could penalize
  performance.

- According to [Wikipedia](https://en.wikipedia.org/wiki/Exception_safety), at
  least basic exception safety is required to write robust code.

## The rule of five: move constructor and move assignment operator

- The `Wallet` class, in its current shape, follows the rule of three. It is
  actually pretty complete already. But since C++11, we can define two more
  functions, namely move constructor and move assignment operator, to utilize
  the latest feature of C++. This is called
  [the rule of five](https://en.cppreference.com/w/cpp/language/rule_of_three).

    - Note that a copy constructor will be implicitly defined if we don't
      prepare
      one ourselves. For a move constructor, it will only be implicitly defined
      if all of the following is true:
        - There are no user-declared copy constructors;
        - There are no user-declared copy assignment operators;
        - There are no user-declared move assignment operators;
        - There is no user-declared destructor.
    - So we don't need to be concerned about unexpected results from an
      implicitly
      defined move constructor.

- The use of move constructor and move assignment operator can be demonstrated
  in the following snippet:

  ```C++
  void start_new_thread() {
    Wallet myWallet = Wallet(2147483647);
    myThread = std::thread([](Wallet w) {
      // Long-running task
    }, myWallet);
    myThread.detach();
  }
  ```

- The function tries to pass a HUGE object to a thread, which should work but
  would be very slow as each time the copy constructor is called to allocate
  memory for a brand new object. We can't pass the object by reference by
  simply changing `Wallet w` to `Wallet &w` as `myWallet` will be out of scope
  immediately after `myThread.detach()`.

    - There are definitely ways to circumvent this dilemma. For example, we make
      `Wallet myWallet` a global object. But they are not as natural.

- C++11 introduces the so-called move semantics that can achieve this. Examining
  the above code, we notice a pattern: while `w` needs to be constructed and new
  memory will be allocated, `myWallet` will go out of scope very soon and memory
  it manages will need to be deallocated. Why not just "hand over" the
  internal buffer of `myWallet` to `w`? We can eliminate one unnecessary
  allocation and one unnecessary deallocation by doing this.
    - This "handing over" is implemented as move constructor and move assignment
      operator:
  ```C++
  void start_new_thread() {
    Wallet myWallet = Wallet(2147483647);
    myThread = std::thread([](Wallet &&w) {
      // Long-running task
    }, std::move(myWallet));
    myThread.detach();
    // Can't access myWallet now as its internal buffer has been handed over to
    // other objects.
    assert(myWallet.ptr0 == nullptr);
    assert(myWallet.ptr1 == nullptr);
  }
  ```

## A better solution

- The above practice is mainly used to demonstrate the peculiarity of the
  interplay between C and C++.
    - If pointers are really needed in an RAII-enabled class, it is better
      to go the "C++" way--instead of using raw pointers, we should use smart
      pointers instead.
    - Smart pointer has its own RAII wrapper, which could, hopefully, handle
      the edge cases for us "automatically".

## References

- [CPP Reference - smart pointers](https://en.cppreference.com/book/intro/smart_pointers)
- [Standard C++ Foundation - How should I handle resources if my constructors may throw exceptions?](https://isocpp.org/wiki/faq/exceptions#selfcleaning-members)
- [StackOverflow - What's the difference between assignment operator and copy constructor?](https://stackoverflow.com/questions/11706040/whats-the-difference-between-assignment-operator-and-copy-constructor)
- [Back to Basics: RAII and the Rule of Zero - Arthur O'Dwyer - CppCon 2019](https://www.youtube.com/watch?v=7Qgd9B1KuMQ)
