#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
  char welcome_msg[] = "Please type something and press Enter! (only first 20 characters will be recorded)\n";
  write(1, welcome_msg, sizeof(welcome_msg)/sizeof(welcome_msg[0]));
  char in_buf[] = "This is the string being write()'ed to stdin\n";
  char out_buf[30] = {0};
  write(STDIN_FILENO, in_buf, sizeof(in_buf)/sizeof(in_buf[0]));
  
  read(0, out_buf, 30); // Nope...this won't work
  out_buf[sizeof(out_buf)/sizeof(out_buf[0])] = '\0';
  write(1, "You typed:\n", 11);
  write(2, out_buf, 20);
  return 0;
}