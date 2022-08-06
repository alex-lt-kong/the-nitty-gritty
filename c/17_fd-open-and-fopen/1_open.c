#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
  int fd = open("/tmp/test.txt", O_CREAT | O_WRONLY);
  if (fd == -1) {
    fprintf(stderr, "Can't open the file!\n");
    return -1;
  }
  printf("fd == %d, the expected value is 3\n", fd);
  char buf[] = "writing():\nHello world!\n";
  write(fd, buf, sizeof(buf)/sizeof(buf[0]));
  close(fd);

  // fd == 1 means stdout and 2 means stderr
  write(1, buf, sizeof(buf)/sizeof(buf[0]));
  write(2, buf, sizeof(buf)/sizeof(buf[0]));
  return 0;
}