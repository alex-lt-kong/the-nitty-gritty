#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
    int n;
    char buf[128] = {0};
    int fd = open("/etc/bash.bashrc", O_RDONLY);
    n = read(fd, buf, sizeof(buf)/sizeof(buf[0]) - 1);
    close(fd);
    printf("%s\n", buf);
    return 0;
}