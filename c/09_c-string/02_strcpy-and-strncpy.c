#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUF_SIZE 4096

int main() {
  char msg0[BUF_SIZE];
  char msg_comp0[] = "Hello ";
  char msg_comp1[] = "world";
  char msg_comp2[] = "!";
  char* msg1 = malloc(strlen(msg_comp0) + strlen(msg_comp1) + strlen(msg_comp2) + 1);

  /*
  strncpy() is NOT a "safe" version of strcpy(). It is defined to:
  * copies the first num characters of source to destination.
  * If the end of the source C string (which is signaled by a null-character) is found before num characters have been
    copied,destination is padded with zeros until a total of num characters have been written to it.
  * No null-character is implicitly appended at the end of destination if source is longer than num. Thus, in this case,
    destination shall not be considered a null terminated C string (reading it as such would overflow).

  */

  printf("msg_comp0: [%s]\nmsg_comp1: [%s]\nmsg_comp3: [%s]\n", msg_comp0, msg_comp1, msg_comp2);
  strcpy(msg0 + strnlen(msg0, BUF_SIZE), msg_comp0);
  strcpy(msg0 + strnlen(msg0, BUF_SIZE), msg_comp1);
  strcpy(msg0 + strnlen(msg0, BUF_SIZE), msg_comp2);
  printf("msg0: [%s]\n", msg0);
  /* copy to sized buffer (overflow safe): */
  strncpy(msg1 + strnlen(msg1, BUF_SIZE), msg_comp0, BUF_SIZE - 1);
  strncpy(msg1 + strnlen(msg1, BUF_SIZE), msg_comp1, BUF_SIZE - 1);
  strncpy(msg1 + strnlen(msg1, BUF_SIZE), msg_comp2, BUF_SIZE - 1);
  printf("msg1: [%s]\n", msg1);
  return 0;
}