#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <new>
#include <stdexcept>

using namespace std;

class Wallet {
private:
  void mallocPtrs() {
    // This is a prime opportunity to try realloc() but for the sake of
    // simplicity, it is not used
    ptr0 = (int *)malloc(sizeof(int) * wallet_size);
    if (ptr0 == nullptr) {
      cout << "1st malloc() failed\n";
      throw bad_alloc();
    }
    ptr1 = (int *)malloc(sizeof(int) * wallet_size);
    if (ptr1 == nullptr) {
      cout << "2nd malloc() failed\n";
      // Need to free() the already malloc()'ed pointer manually
      // as destructor will NOT be call if exception is thrown in
      // constructor
      free(ptr0);
      // Also need to set curr_ptr0 to nullptr as mallocPtrs() could be
      // called by functions other than constructors. When it is called by
      // another functions and an exception is thrown, destructor will be called
      // we we risk double free()ing curr_ptr0 if it is not set to nullptr.
      ptr0 = nullptr;
      wallet_size = -1;
      throw bad_alloc();
    }
  }

public:
  // In order to check the internal state of the class, we expose all member
  // variables to users
  int *ptr0 = nullptr;
  int *ptr1 = nullptr;
  ssize_t wallet_size = -1;
  Wallet(ssize_t wallet_size) {
    if (wallet_size <= 0) {
      // man malloc: If size is 0, then malloc() returns either NULL, or a
      // unique pointer value that can later be successfully passed to free().
      // To avoid confusion and possible issue with memcpy(), etc., let's just
      // ban this
      throw invalid_argument("wallet_size must be positive");
    }
    cout << "DynamicWallet (" << wallet_size << ") called" << endl;
    this->wallet_size = wallet_size;
    mallocPtrs();
  }

  // Copy constructor: if *this instance is NOT initialized and we want to
  // initialize *this object with an existing object, copy constructor will be
  // called.
  Wallet(const Wallet &rhs) {
    wallet_size = rhs.wallet_size;
    mallocPtrs();
    // Wait, rhs.curr_ptr0 is a private member of rhs, how can we access it?
    // Answer: copy constructor is a "member function" of the class which can
    // access all the member variables of instances of the class it's member of.
    memcpy(ptr0, rhs.ptr0, sizeof(int) * wallet_size);
    memcpy(ptr1, rhs.ptr1, sizeof(int) * wallet_size);
    cout << "copy constructor called" << endl;
  }
  // Move constructor
  Wallet(Wallet &&rhs) noexcept {
    wallet_size = rhs.wallet_size;
    ptr0 = rhs.ptr0;
    ptr1 = rhs.ptr1;
    rhs.wallet_size = -1;
    rhs.ptr0 = nullptr;
    rhs.ptr1 = nullptr;
    cout << "move constructor called" << endl;
  }

  // Copy assignment operator: if *this instance is already initialized and we
  // want to replace its content with the content from an existing object,
  // copy assignment operator will be called
  Wallet &operator=(const Wallet &rhs) {
    cout << "copy assignment operator called";
    if (this != &rhs) {                     // not a self-assignment
      if (wallet_size != rhs.wallet_size) { // resource cannot be reused
        // This copy assignment operator only provides basic exception
        // guarantee--while it hopefully leaks no memory whatsoever, *this
        // object will NOT be "rolled back" if mallocPtrs() throws an exception.
        cout << " and wallet resized from " << wallet_size << " to "
             << rhs.wallet_size;
        this->~Wallet();
        wallet_size = rhs.wallet_size;
        // Using realloc() may further improve performance, but this is not
        // the purpose of this PoC and thus it is not used.
        mallocPtrs();
      } else {
        cout << " and previous memory reused for new data";
      }
      memcpy(ptr0, rhs.ptr0, sizeof(int) * wallet_size);
      memcpy(ptr1, rhs.ptr1, sizeof(int) * wallet_size);

    } else {
      cout << " but it is a self-assignment";
    }
    cout << endl;
    return *this;
  }
  // Move assignment operator
  Wallet &operator=(Wallet &&rhs) {
    cout << "move assignment operator called";
    if (this != &rhs) {
      this->~Wallet();
      wallet_size = rhs.wallet_size;
      ptr0 = rhs.ptr0;
      ptr1 = rhs.ptr1;
      rhs.ptr0 = nullptr;
      rhs.ptr1 = nullptr;
      rhs.wallet_size = -1;
      cout << endl;
    } else {
      cout << " but it is a self-assignment" << endl;
    }
    return *this;
  }

  int &operator()(size_t i, size_t j) {
    if (i > 1 || (ssize_t)j >= wallet_size) {
      throw out_of_range("");
    }
    return (i == 0 ? ptr0[j] : ptr1[j]);
  }
  // If copy constructor is explicitly defined, move constructor will NOT be
  // implicitly defined. The same applies to copy assignment operator and
  // move assignment operator.

  ~Wallet() noexcept {
    free(ptr0);
    free(ptr1);
  }
};

int main() {
  cout << "* Basic operation\n";
  auto first_wallet = Wallet(2048);
  first_wallet(0, 0) = 3;
  first_wallet(1, 2047) = 666;
  assert(first_wallet(0, 0) == 3);
  assert(first_wallet(1, 2047) == 666);
  cout << "Okay\n" << endl;

  cout << "* Trying copy constructor\n";
  auto second_wallet = first_wallet;
  second_wallet(0, 0) = 31415926;
  second_wallet(1, 2047) = -1;
  // As second_wallet is a copy of first_wallet, they should have their own
  // internal data buffer
  assert(first_wallet(0, 0) == 3);
  assert(first_wallet(1, 2047) == 666);
  assert(second_wallet(0, 0) == 31415926);
  assert(second_wallet(1, 2047) == -1);
  cout << "Okay\n" << endl;

  cout << "* Trying copy assignment opeartor with different wallet size\n";
  auto third_dwallet = Wallet(1);
  third_dwallet = first_wallet;
  third_dwallet(0, 0) = -123;

  // first_wallet/second_wallet/third_wallet should have different internal data
  // buffer, changing any one of them shouldn't impact the rest of them
  assert(first_wallet(0, 0) == 3);
  assert(first_wallet(1, 2047) == 666);
  assert(second_wallet(0, 0) == 31415926);
  assert(second_wallet(1, 2047) == -1);
  assert(third_dwallet(0, 0) == -123);
  assert(third_dwallet(1, 2047) == 666);
  cout << "Okay\n" << endl;

  cout << "* Trying copy assignment opeartor with different wallet size\n";
  auto fourth_wallet = Wallet(2048);
  fourth_wallet = first_wallet;
  fourth_wallet(0, 0) = -9527;
  fourth_wallet(1, 2047) = 16888;
  fourth_wallet(1, 765) = 2147483647;

  assert(first_wallet(0, 0) == 3);
  assert(first_wallet(1, 2047) == 666);
  assert(fourth_wallet(0, 0) == -9527);
  assert(fourth_wallet(1, 765) == 2147483647);
  assert(fourth_wallet(1, 2047) == 16888);
  cout << "Okay\n" << endl;

  cout << "* Trying self-assignment\n";
  first_wallet = first_wallet;

  assert(first_wallet(0, 0) == 3);
  assert(first_wallet(1, 2047) == 666);
  cout << "Okay\n" << endl;

  cout << "* Trying move constructor\n";
  Wallet fifth_wallet = first_wallet;
  fifth_wallet(1, 2) = 468;
  assert(fifth_wallet(0, 0) == 3);
  assert(fifth_wallet(1, 2047) == 666);
  assert(fifth_wallet(1, 2) == 468);
  Wallet sixth_wallet = std::move(fifth_wallet);
  assert(sixth_wallet(0, 0) == 3);
  assert(sixth_wallet(1, 2047) == 666);
  assert(sixth_wallet(1, 2) == 468);
  // fifth_wallet's ownership is gone, its internal buffer should always be NULL
  assert(fifth_wallet.ptr0 == nullptr);
  assert(fifth_wallet.ptr1 == nullptr);
  assert(fifth_wallet.wallet_size == -1);
  cout << "Okay\n" << endl;

  cout << "* Trying move assignment operator\n";
  sixth_wallet = std::move(second_wallet);
  sixth_wallet(1, 2) = 469;
  assert(sixth_wallet(0, 0) == 31415926);
  assert(sixth_wallet(1, 2) == 469);
  assert(sixth_wallet(1, 2047) == -1);
  assert(second_wallet.ptr0 == nullptr);
  assert(second_wallet.ptr1 == nullptr);
  assert(second_wallet.wallet_size == -1);
  cout << "Okay\n" << endl;

  // By default, Linux uses opportunistic memory allocation, so that successful
  // malloc() may still fail and get killed later. This also obfuscate the
  // behavior we want to observe. Need to turn it off by issuing:
  // echo 2 > /proc/sys/vm/overcommit_memory
  // Reference:
  // https://stackoverflow.com/questions/16674370/why-does-malloc-or-new-never-return-null
  // This option breaks AddressSanitizer, but Valgrind still works:
  //  valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
  //  --verbose --log-file=valgrind-out.txt ./build/02_raw-pointers-with-raii
  cout << "* Trying to fail only the 2nd malloc()\n";
  try {
    ssize_t wallet_size = INT_MAX / 2;
    auto seventh_wallet = Wallet(wallet_size);
    seventh_wallet(0, 234) = 987;
    seventh_wallet(0, wallet_size - 1) = -987;
    seventh_wallet(1, 2) = 123;
    seventh_wallet(1, wallet_size - 1) = 213124;
    assert(seventh_wallet(0, 234) == 987);
    assert(seventh_wallet(0, wallet_size - 1) == -987);
    assert(seventh_wallet(1, 2) == 123);
    assert(seventh_wallet(1, wallet_size - 1) == 213124);
  } catch (bad_alloc const &) {
    cout << "Okay: bad_alloc caught as expected, but you need to check if ONLY "
            "the 2nd malloc() failed.\n"
         << endl;
  }

  return 0;
}
