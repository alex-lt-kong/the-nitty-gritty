#include <stdio.h>
#include <string.h>

#define MSG_BUF_SIZE 4096

int main() {
  char msg[MSG_BUF_SIZE];
  int a = 12345;
  char text[] = "Hello world!";
  printf("Printf %s %d\n");
  snprintf(msg, MSG_BUF_SIZE, "snprintf() with format chars [%d] [%s]");
  printf("text: %s\na: %d\n", text, a);
  printf("msg: %s\n", msg);
  return 0;
}