// #include <print>
#include <stdio.h>
#include <stdlib.h>

void split_address(void *ptr, int page_size) {
  unsigned long address = (unsigned long)ptr;  // Cast pointer to unsigned long
  int offset_bits = __builtin_ctz(page_size);  // Calculate bits for offset
  unsigned long page_offset = address & ((1UL << offset_bits) - 1);  // Mask offset bits
  unsigned long page_index = address >> offset_bits;  // Shift out offset bits

  printf("Address: %p\n", ptr);
  printf("Page Index: 0x%lx\n", page_index);
  printf("Page Offset: 0x%lx\n", page_offset);
}

int main() {
  int data = 42;  // Example variable
  split_address(&data, 4096);  // Pass the pointer to the function
  return 0;
}