# Resource acquisition is initialization

* RAII is an awkward name for a useful feature. One can understand it as C++'s
answer to a `finally` keyword in Python, etc.

* Let's consider this Python snippet:

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

    * `finally` is needed because we don't want `conn` to be left open whether
    or not an exception is thrown.


* In comparison, RAII is rather straightforward, let's wrap the resource releasing
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

    * The "official" opinion from
    [Bjarne Stroustrup](https://www.stroustrup.com/bs_faq2.html#finally)
    also argues that RAII is almost always better than a "finally" keyword.

    * Theoretically, RAII means "to bind the life cycle of a
    resource that must be acquired before use (allocated heap memory, thread
    of execution, open socket, open file, locked mutex, disk space,
    database connection—anything that exists in limited supply) to the lifetime
    of an object. ".
        * Microsoft summarizes the RAII principle as "to give ownership of any
        heap-allocated resource—for example, dynamically-allocated memory or
        system object handles—to a stack-allocated object whose destructor
        contains the code to delete or free the resource and also any
        associated cleanup code."

    * To be specific, RAII encapsulates each resource into a class, where:
        * the constructor acquires the resource and establishes all class
        invariants or throws an exception if that cannot be done,
        * the destructor releases the resource and never throws exceptions; 


* What if constructors or destructors throw exceptions? Things get a bit
messier here.

    * In case of exceptions thrown by a constructor, the object’s destructor is
    *not* invoked. If your object has already done something that needs to be
    undone (such as allocating some memory, opening a file, or locking a
    semaphore), this "stuff that needs to be undone" must be manually released
    in the constructor's error handling code.
    * But why the destructor is not called if an exception is thrown within
    a constructor? In C++, the lifecycle of an object starts when its
    initialization is done (roughly the same as "constructor returns
    successfully"). If an exception is raised in its constructor, the object's
    lifecycle has not started yet, so its destructor will not be called.
    * In case of destructors--[destructors should never fail](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Re-never-fail). If
    some function calls in a destructor could possibly throw exceptions, we
    should handle the exception within destructor, rather than propagating
    the exception up the call stack.

* Apart from being handy for database connection management, etc
RAII is also the underlying principle of [smart pointers](../10_smart-pointers).
Say we use a class to wrap a raw pointer, using constructor to `malloc()`
memory and use its destructor to `free()` the memory--brilliant! a
"smart pointer" already!

    * But sure, the issue is a bit more complicated than this. The above attempt
    only works if one heap memory object has only one pointer pointing to it;
    otherwise other pointers will end up pointing to nowhere, causing
    unexpected behaviors.

## Pointers with RAII and [the rule of three](https://en.cppreference.com/w/cpp/language/rule_of_three)

* So far, so good--playing in the C++ playground, things are nice and tidy.
However, as always, it becomes more interesting (or horrible if you wish) when
C comes into play.

* What if, for whatever reason, we need to to `malloc()` raw pointers in a
constructor and `free()` them in destructors? Going down this rabbit hole,
things start to turn surreal.

* Think about this naive implementation. Memory is `malloc()`ed in constructor
and `free()`ed in destructor, perfect RAII:

    ```C++
    class Wallet {
    private:
        void mallocPtrs() {
            this->curr_ptr0 = (int*)malloc(sizeof(int) * ARR_SIZE);
            this->curr_ptr1 = (int*)malloc(sizeof(int) * ARR_SIZE);
        }
    public:
        int* curr_ptr0;
        int* curr_ptr1;

        Wallet () {
            printf("Wallet () called\n");
            mallocPtrs();
        }
        ~Wallet () {
            free(this->curr_ptr0);
            free(this->curr_ptr1);
        }

    };
    ```
    * The code should work 99.9% of time if both `malloc()`s do not fail. But
    what if *both* of them fail? It means `curr_ptr0` and `curr_ptr1` will be
    both `NULL`, and any subsequent dereference could cause unexpected
    result! We need to prevent this.

* A slightly better version. Now we throw `bad_alloc` exception if `malloc()`
fails. This should prevent NULL pointers dereference:

    ```C++
    class Wallet {
    private:
        void mallocPtrs() {
            this->curr_ptr0 = (int*)malloc(sizeof(int) * ARR_SIZE);
            if (this->curr_ptr0 == NULL) {
                std::bad_alloc exception;
                throw exception;
            }
            this->curr_ptr1 = (int*)malloc(sizeof(int) * ARR_SIZE);
            if (this->curr_ptr1== NULL) {
                std::bad_alloc exception;
                throw exception;
            }
        }
    public:
        int* curr_ptr0;
        int* curr_ptr1;

        Wallet () {
            printf("Wallet () called\n");
            mallocPtrs();
        }
        ~Wallet () {
            free(this->curr_ptr0);
            free(this->curr_ptr1);
        }
    };
    ```

    * Well yes, it doesn't suffer from NULL pointer dereference. Awesome!
    What if the first `malloc()` succeeds and the second `malloc()`
    fails? A `bad_alloc` exception will be thrown, RAII's principle is followed.
    But wait, what happens to the large amount of memory `malloc()`ed for and
    pointed by `curr_ptr0`? No one takes care of it. It will be left on the heap
    lonely, forever! No, we need to handle this as well.

* Another version is prepared to handle this. Now we will manually `free()`
the memory `malloc()`ed for the 1st pointer in case only the 2nd `malloc()`
fails:

    ```C++
    class Wallet {
    private:
        void mallocPtrs() {
            this->curr_ptr0 = (int*)malloc(sizeof(int) * ARR_SIZE);
            if (this->curr_ptr0 == NULL) {
                std::bad_alloc exception;
                throw exception;
            }
            this->curr_ptr1 = (int*)malloc(sizeof(int) * ARR_SIZE);
            if (this->curr_ptr1== NULL) {
                // handle the already malloc()'ed pointer manually.
                free(this->curr_ptr0);
                std::bad_alloc exception;
                throw exception;
            }
        }
    public:
        int* curr_ptr0;
        int* curr_ptr1;

        Wallet () {
            printf("Wallet () called\n");
            mallocPtrs();
        }
        ~Wallet () {
            free(this->curr_ptr0);
            free(this->curr_ptr1);
        }
    };
    ```
* We are done? Not yet! Think about this common usage:

    ```C++
    Wallet first_wallet = Wallet();
    first_wallet.curr_ptr0[0] = 1;
    first_wallet.curr_ptr1[2] = 2147483647;
    Wallet second_wallet = first_wallet;
    second_wallet.curr_ptr0[0] = 31415926;
    second_wallet.curr_ptr1[2] = -1;
    ```
    * As we don't define a copy constructor, the compiler will prepare one
    for us. However, compiler doesn't know if we want to copy `curr_ptr0` and
    `curr_ptr1` by reference or by value. But as we have two integer pointers,
    the default copy constructor will copy only the value of these pointers.
    That is, it is copy by reference.
    * The most singificant implication is that two seemingly distinct objects
    will share the same buffers. Meaning that changing one object's value
    will impact the other's.
    * But it is more than this, running the above code result in something
    like below:

    ```C++
    Wallet () called
    first_wallet: 1, 2147483647
    first_wallet: 31415926, -1
    second_wallet: 31415926, -1
    Segmentation fault
    ```
    * Well we accept that there will be buffer sharing, but WTH is there a
    Segfault?? It has to do with C++'s automatic memory management. When an
    object goes out of scope, its destructor is automatically called. In the
    above sample, `first_wallet` and `second_wallet` go out of scope one
    after another and their destructor being called. This is what RAII requires
    as well.
    * After `first_wallet` goes out of scope, we rightfully `free()` both
    pointers. But wait, what happens when `second_wallet`'s destructor is
    called? `curr_ptr0` and `curr_ptr1` are being `free()`ed again, it's
    a [double free](https://encyclopedia.kaspersky.com/glossary/double-free/)!
    Nonono, this shouldn't happen. We need to prepare a
    [copy constructor](./constructor) for it.
    * Apart from the above, we also need to prepare a copy assignment operator.

* A much more robust version is like below. This is something known as
[the rule of three](https://en.cppreference.com/w/cpp/language/rule_of_three):

    ```C++
    class Wallet {
    private:
        void mallocPtrs() {
            this->curr_ptr0 = (int*)malloc(sizeof(int) * ARR_SIZE);
            if (this->curr_ptr0 == NULL) {
                std::bad_alloc exception;
                throw exception;
            }
            this->curr_ptr1 = (int*)malloc(sizeof(int) * ARR_SIZE);
            if (this->curr_ptr1== NULL) {
                free(this->curr_ptr0);
                // Need to handle the already malloc()'ed pointer manually.
                std::bad_alloc exception;
                throw exception;
            }
        }
    public:
        int* curr_ptr0;
        int* curr_ptr1;

        Wallet () {
            printf("Wallet () called\n");
            mallocPtrs();
        }
        // Copy constructor
        Wallet(const Wallet& rhs) {
            mallocPtrs();  
            memcpy(this->curr_ptr0, rhs.curr_ptr0, sizeof(int) * ARR_SIZE);
            memcpy(this->curr_ptr1, rhs.curr_ptr1, sizeof(int) * ARR_SIZE);
            printf("copy Wallet() called\n");
        }
        // Copy assignment operator
        Wallet& operator=(const Wallet& rhs) {
            memcpy(this->curr_ptr0, rhs.curr_ptr0, sizeof(int) * ARR_SIZE);
            memcpy(this->curr_ptr1, rhs.curr_ptr1, sizeof(int) * ARR_SIZE);
            printf("operator=(const Wallet& rhs) called\n");
            return *this;
        }
        ~Wallet () {
            free(this->curr_ptr0);
            free(this->curr_ptr1);
        }
    };
    ```
    * The difference between a copy constructor and a copy assignment operator
    is that "a copy constructor is used to initialize a previously
    UNinitialized object from some other object's data. An assignment
    operator is used to replace the data of a previously INitialized object
    with some other object's data. "

* The above sample works fine, but it still hides some important complexity
because `ARR_SIZE` is something predefined and fixed, so that we can always
re-use existing `malloc()`ed memory. What if `ARR_SIZE` is dynamic? It makes
copy constructor and copy assignment operator much more complicated.

    ```C++

    class DynamicWallet {
    private:
        void mallocPtrs() {
            curr_ptr0 = (int*)malloc(sizeof(int) * wallet_size);
            if (curr_ptr0 == NULL) {
                std::bad_alloc exception;
                throw exception;
            }
            curr_ptr1 = (int*)malloc(sizeof(int) * wallet_size);
            if (curr_ptr1== NULL) {
                free(curr_ptr0);
                // Need to handle the already malloc()'ed pointer manually.
                std::bad_alloc exception;
                throw exception;
            }
        }
    public:
        int* curr_ptr0;
        int* curr_ptr1;
        size_t  wallet_size;
        DynamicWallet (size_t wallet_size) {
            cout << "DynamicWallet () called" << endl;
            this->wallet_size = wallet_size;
            mallocPtrs();
        }
        // Copy constructor
        DynamicWallet(const DynamicWallet& rhs) {
            wallet_size = rhs.wallet_size;
            mallocPtrs();
            memcpy(curr_ptr0, rhs.curr_ptr0, sizeof(int) * wallet_size);
            memcpy(curr_ptr1, rhs.curr_ptr1, sizeof(int) * wallet_size);
            cout << "copy DynamicWallet() called" << endl;
        }
        // Copy assignment operator
        DynamicWallet& operator=(const DynamicWallet& rhs) {
            this->wallet_size = rhs.wallet_size;
            free(curr_ptr0);
            free(curr_ptr1);
            mallocPtrs();
            memcpy(curr_ptr0, rhs.curr_ptr0, sizeof(int) * wallet_size);
            memcpy(curr_ptr1, rhs.curr_ptr1, sizeof(int) * wallet_size);
            cout << "operator=(const DynamicWallet& rhs) called" << endl;
            return *this;
        }
        ~DynamicWallet () {
            free(curr_ptr0);
            free(curr_ptr1);
        }
    }
    ```

  * If `wallet_size` is dynamic, we need to `free()` previously `malloc()`ed
  memory and then `malloc()` again.

  * But is the above example good enough? Unfortunately, the answer is no.
  What could go wrong if we do the following?:

    ```C++
        DynamicWallet first_dwallet = DynamicWallet(2048);
        first_dwallet.curr_ptr0[0] = 3;
        first_dwallet.curr_ptr1[2047] = 666;
        first_dwallet = first_dwallet
    ```

* A version that corrects this bug can be found [here](./pointer.cpp)

### A better solution

* The above practice is mainly used to demonstrate the peculiarity of the
interplay between C and C++.
  * If pointers are really needed in an RAII-enabled class, it is better
  to go the "C++" way--instead of using raw pointers, we should use smart
  pointers instead.
  * Smart pointer has its own RAII wrapper, which could, hopefully, handle
  the edge cases for us "automatically".

## References

* [Microsoft - Smart pointers (Modern C++)](https://learn.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=msvc-170)
* [CPP Reference - smart pointers](https://en.cppreference.com/book/intro/smart_pointers)
* [Standard C++ Foundation - How should I handle resources if my constructors may throw exceptions?](https://isocpp.org/wiki/faq/exceptions#selfcleaning-members)
* [StackOverflow - What's the difference between assignment operator and copy constructor?](https://stackoverflow.com/questions/11706040/whats-the-difference-between-assignment-operator-and-copy-constructor)
* [Back to Basics: RAII and the Rule of Zero - Arthur O'Dwyer - CppCon 2019](https://www.youtube.com/watch?v=7Qgd9B1KuMQ)
