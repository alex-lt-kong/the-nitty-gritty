// #include <print>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

void split_address(void *ptr, long page_size) {
  unsigned long address = (unsigned long)ptr;  // Cast pointer to unsigned long
  int offset_bits = __builtin_ctz(page_size);  // Calculate bits for offset
  unsigned long page_offset = address & ((1UL << offset_bits) - 1);  // Mask offset bits
  unsigned long page_index = address >> offset_bits;  // Shift out offset bits

  printf("Address: %p\n", ptr);
  printf("Page Index: 0x%lx\n", page_index);
  printf("Page Offset: 0x%lx\n", page_offset);
}

int main() {
  int data = 42;
  const long page_size = sysconf(_SC_PAGESIZE);
  printf("page_size: %ld\n", page_size);
  printf("%ld\n", (long)(&data) % page_size);
  split_address(&data, page_size);
  return 0;
}