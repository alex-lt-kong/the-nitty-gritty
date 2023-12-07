#include <iostream>
#include <vector>

void notMoving(std::string &&rhs) {
  std::string s = rhs;
  std::cout << "s@notMoving(): " << s << std::endl;
}
void moving(std::string &&rhs) {
  std::string s = std::move(rhs);
  std::cout << "s@moving(): " << s << std::endl;
}

void test1_MoveMayNotMoveAnything() {
  std::cout << "test1_MoveMayNotMoveAnything:\n";
  std::string s = "Hello world!";
  std::cout << "s is: " << s << std::endl;
  notMoving(std::move(s));
  std::cout << "s is: " << s << std::endl;
  moving(std::move(s));
  std::cout << "s is: " << s << std::endl;
}

void overloaded(int const &arg) { std::cout << "by lvalue\n"; }
void overloaded(int &&arg) { std::cout << "by rvalue\n"; }

template <typename t> void forwarding(t &&arg) {
  std::cout << "via std::forward: ";
  overloaded(std::forward<t>(arg));
  std::cout << "via std::move: ";
  overloaded(std::move(arg)); // conceptually this would invalidate arg
  std::cout << "by simple passing: ";
  overloaded(arg);
}

void test2_MoveVsForward() {
  std::cout << "test2_MoveVsForward:\n";
  std::cout << "--- initial caller passes rvalue ---\n";
  forwarding(5);
  std::cout << "--- initial caller passes lvalue ---\n";
  int x = 5;
  // forwarding(x) only works if forwarding() is a function template instead of
  // a concrete function.
  forwarding(x);
}

int main(void) {
  test1_MoveMayNotMoveAnything();
  std::cout << "\n";
  test2_MoveVsForward();
  std::cout << "\n";
  return 0;
}
