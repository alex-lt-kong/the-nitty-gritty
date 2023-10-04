#include <iostream>
#include <memory>

using namespace std;

void dumb_pointer_with_leak() {
  cout << "dumb_pointer_with_leak() called\n";
  int *valuePtr = new int(15);
  int x = 45;

  if (x == 45) {
    cout << "dumb_pointer_with_leak() returned\n" << endl;
    return; // here we have a memory leak, valuePtr is not deleted
  }
  delete valuePtr;
}

void create_unique_ptr() {
  cout << "create_unique_ptr() called\n";
  unique_ptr<int> valuePtr(new int(15));
  int x = 45;

  if (x == 45) {
    cout << "create_unique_ptr() returned\n" << endl;
    // no memory leak anymore as unique_ptr's destructor is called to release
    // the recource.
    return;
  }
}

void unique_ptr_converted_from_dumb_ptr(size_t arr_size) {
  cout << "unique_ptr_converted_from_dumb_ptr() called\n";
  struct FreeDeleter {
    void operator()(void *p) const { std::free(p); }
  };

  int *dynamic_int_arr = (int *)malloc(sizeof(int) * arr_size);
  // In reality, malloc() should be some deep-rooted C functions,
  // probably in a compiled so file. Here we simplify the scenario by
  // directly using a malloc()
  for (int i = 0; i < arr_size; i++) {
    dynamic_int_arr[i] = i;
  }
  unique_ptr<int[], FreeDeleter> smart_int_ptr(dynamic_int_arr);

  for (size_t i = 0; i < arr_size; ++i) {
    cout << smart_int_ptr[i] << std::endl;
  }
  cout << "unique_ptr_converted_from_dumb_ptr() returned\n" << endl;
}

void callee_func_raw_ptr(int *arg) {
  cout << "callee_func_raw_ptr() called\n";
  cout << "Value of a: " << *arg << endl;
  ++(*arg);
  cout << "Value of a: " << *arg << endl;
  cout << "callee_func_raw_ptr() returned" << endl;
}

void callee_func_ref(unique_ptr<int> &arg) {
  cout << "callee_func_ref() called\n";
  cout << "Value of a: " << *arg << endl;
  ++(*arg);
  cout << "Value of a: " << *arg << endl;
  cout << "callee_func_ref() returned" << endl;
}

void callee_func_move(unique_ptr<int> arg) {
  cout << "callee_func_move() called\n";
  cout << "Value of a: " << *arg << endl;
  ++(*arg);
  cout << "Value of a: " << *arg << endl;
  cout << "callee_func_move() returned" << endl;
}

void pass_unique_ptr_to_func() {
  cout << "pass_unique_ptr_to_func() called\n";
  unique_ptr<int> x(new int(0));
  *x = 45;
  callee_func_ref(x);
  cout << "x is: " << *x << endl;
  callee_func_raw_ptr(x.get());
  cout << "x is: " << *x << endl;
  callee_func_move(move(x));
  cout << "x.get() == nullptr: " << (x.get() == nullptr) << endl;
  cout << "pass_unique_ptr_to_func() returned\n";
}

int main() {
  dumb_pointer_with_leak();
  create_unique_ptr();
  unique_ptr_converted_from_dumb_ptr(12);
  pass_unique_ptr_to_func();
  return 0;
}
