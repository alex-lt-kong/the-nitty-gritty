#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
 #include <unistd.h>

int main() {
    int n;
    char buf[100];
    int fd = open("/etc/passwd", O_RDONLY);
    n = read(fd, buf, 100);
    close(fd);
    return 0;
}