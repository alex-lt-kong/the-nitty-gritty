#include <iostream>

using namespace std;

template <typename T> void func1(T a) {
  ++a;
  cout << a << endl;
}

template <typename T> void func2(T &a) {
  ++a;
  cout << a << endl;
}

template <typename T> void func3(const T &a) {
  // ++a; // error: Variable 'a' declared const here
  cout << a << endl;
}

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