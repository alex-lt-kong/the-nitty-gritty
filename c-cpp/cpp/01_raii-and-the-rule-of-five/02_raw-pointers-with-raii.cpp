#include <iostream>
#include <new>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARR_SIZE 65536

using namespace std;

class Wallet {
private:
  int *ptr0 = nullptr;
  int *ptr1 = nullptr;
  size_t wallet_size = 0;

  void freePtrs() noexcept {
    free(ptr0);
    free(ptr1);
  }

  void mallocPtrs() {

    if (wallet_size == 0) {
      // man malloc: If size is 0, then malloc() returns either NULL, or a
      // unique pointer value that can later be successfully passed to free().
      // To avoid confusion and possible issue with memcpy(), etc., let's just
      // ban this
      throw invalid_argument("wallet_size must be positive");
    }
    ptr0 = (int *)malloc(sizeof(int) * wallet_size);
    if (ptr0 == nullptr) {
      throw bad_alloc();
    }
    ptr1 = (int *)malloc(sizeof(int) * wallet_size);
    if (ptr1 == nullptr) {
      // Need to free() the already malloc()'ed pointer manually
      // as destructor will NOT be call if exception is thrown in
      // constructor
      free(ptr0);
      // Also need to set curr_ptr0 to nullptr as mallocPtrs() could be
      // called by functions other than constructors. When it is called by
      // another functions and an exception is thrown, destructor will be called
      // we we risk double free()ing curr_ptr0 if it is not set to nullptr.
      ptr0 = nullptr;
      throw bad_alloc();
    }
  }

public:
  Wallet(size_t wallet_size) {

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
        cout << " and wallet resized";
        freePtrs();
        wallet_size = rhs.wallet_size;
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

  int &operator()(size_t i, size_t j) {
    if (i > 1 || j >= wallet_size) {
      throw out_of_range("");
    }
    return (i == 0 ? ptr0[j] : ptr1[j]);
  }
  // If copy constructor is explicitly defined, move constructor will NOT be
  // implicitly defined. The same applies to copy assignment operator and
  // move assignment operator.

  ~Wallet() noexcept { freePtrs(); }
};

int main() {
  cout << "* Basic operation" << endl;
  Wallet first_dwallet = Wallet(2048);
  first_dwallet(0, 0) = 3;
  first_dwallet(1, 2047) = 666;
  cout << "first_dwallet: " << first_dwallet(0, 0) << ", "
       << first_dwallet(1, 2047) << endl
       << endl;

  cout << "* Trying copy constructor" << endl;
  Wallet second_dwallet = first_dwallet;
  second_dwallet(0, 0) = 31415926;
  second_dwallet(1, 2047) = -1;
  cout << "first_dwallet: " << first_dwallet(0, 0) << ", "
       << first_dwallet(1, 2047) << endl;
  cout << "second_dwallet: " << second_dwallet(0, 0) << ", "
       << second_dwallet(1, 2047) << endl
       << endl;

  cout << "* Trying copy assignment opeartor with different wallet size"
       << endl;
  Wallet third_dwallet = Wallet(1);
  third_dwallet = first_dwallet;
  third_dwallet(0, 0) = -123;

  cout << "first_dwallet: " << first_dwallet(0, 0) << ", "
       << first_dwallet(1, 2047) << endl;
  cout << "second_dwallet: " << second_dwallet(0, 0) << ", "
       << second_dwallet(1, 2047) << endl;
  cout << "third_dwallet: " << third_dwallet(0, 0) << ", "
       << third_dwallet(1, 2047) << endl
       << endl;

  cout << "* Trying copy assignment opeartor with different wallet size"
       << endl;
  Wallet fourth_dwallet = Wallet(2048);
  fourth_dwallet = first_dwallet;
  fourth_dwallet(0, 0) = -9527;
  fourth_dwallet(1, 2047) = 16888;

  cout << "first_dwallet: " << first_dwallet(0, 0) << ", "
       << first_dwallet(1, 2047) << endl;
  cout << "fourth_dwallet: " << fourth_dwallet(0, 0) << ", "
       << fourth_dwallet(1, 2047) << endl
       << endl;

  cout << "* Trying self-assignment" << endl;
  first_dwallet = first_dwallet;
  cout << "first_dwallet: " << first_dwallet(0, 0) << ", "
       << first_dwallet(1, 2047) << endl;
  return 0;
}
