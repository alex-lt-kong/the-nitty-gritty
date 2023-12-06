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

int main(void) {
  std::string s = "Hello world!";
  std::cout << "s is: " << s << std::endl;
  notMoving(std::move(s));
  std::cout << "s is: " << s << std::endl;
  moving(std::move(s));
  std::cout << "s is: " << s << std::endl;
  return 0;
}
