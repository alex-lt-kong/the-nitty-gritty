#include <fcntl.h>
#include <unistd.h>

int main() {
  char welcome_msg[] = "Please type something and press Enter! (only first 20 characters will be recorded)\n";
  write(1, welcome_msg, sizeof(welcome_msg)/sizeof(welcome_msg[0]));
  char buf[20] = {0}; // so that we make sure the input will be null-terminated.
  read(0, buf, 20); // fd==0 means read from stdin
  buf[sizeof(buf)/sizeof(buf[0])] = '\0';
  write(1, "You typed:\n", 11);
  write(2, buf, 20);
  return 0;
}