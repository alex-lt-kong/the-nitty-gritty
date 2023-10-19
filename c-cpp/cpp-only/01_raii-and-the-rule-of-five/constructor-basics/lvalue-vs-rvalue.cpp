#include <iostream>

using namespace std;

// while 5 is an rvalue and a is an lvalue, we make a copy of 5 and assign it to
// a, so it is fine
template <typename T> void func1(T a) {
  ++a;
  cout << a << endl;
}

// a is a reference, it canNOT be bind to a rvalue (i.e., 5), calling it with
// func2(5) will break the compilation.
template <typename T> void func2(T &a) {
  ++a;
  cout << a << endl;
}

// C++ allows us to pass an rvalue to const T &a, but this means a's value can't
// be changed
template <typename T> void func3(const T &a) {
  // ++a; // error: Variable 'a' declared const here
  cout << a << endl;
}

// This is how rvalue reference, e.g., T &&a, comes into play. We can pass
// rvalue to a function and still modify it
template <typename T> void func4(T &&a) {
  ++a;
  cout << a << endl;
}

void func5() {
  string s = "Hello world!";
  (s + s) = s;
}

int main(void) {
  // 5 is an rvalue
  func1(5);
  // func2(5); // error: Candidate function [with T = int] not viable: expects
  // an lvalue for 1st argument
  func3(5);
  func4(5);

  func5();

  return 0;
}