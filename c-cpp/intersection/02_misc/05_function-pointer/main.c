#include <stdio.h>

typedef int (*add_func)(int, int);

int add(int a, int b) { return (a + b); }

void vanilla_func_ptr() {
  printf("vanilla_func_ptr():\n");
  int (*my_add)(int, int);
  my_add = add;
  printf("Address of add(): %p\n", (void *)&add);
  printf("Value of my_add:   %p\n", (void *)my_add);

  int a = 123, b = -1;
  printf("add(): %d, my_add(): %d\n\n", add(a, b), my_add(a, b));
}

void convenient_func_ptr_with_typedef() {
  printf("convenient_func_ptr_with_typedef():\n");
  add_func my_add = add;
  printf("Address of add(): %p\n", (void *)&add);
  printf("Value of my_add:   %p\n", (void *)my_add);

  int a = 314159, b = -1234;
  printf("add(): %d, my_add(): %d\n\n", add(a, b), my_add(a, b));
}

int main() {
  vanilla_func_ptr();
  convenient_func_ptr_with_typedef();
  return 0;
}