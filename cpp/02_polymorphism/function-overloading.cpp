#include "function-overloading.h"

int main() {
  // Function overloading, no vtable needed as everything is fixed at
  // compile time
  my_print(3);
  my_print(3, 4);
  my_print(3, 4, "5", "Hello world", 3.1415);
  my_print();
  return 0;
}