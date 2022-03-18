#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t foo_len(const char *s)
{
  return strlen(s);
}

int main()
{
  const char *a = NULL;
  printf("size of a = %lu\n", foo_len(a));
  // Get the length of an empty char pointer
  return 0;
}