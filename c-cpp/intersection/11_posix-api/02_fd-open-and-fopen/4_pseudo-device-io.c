#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

int main() {
    FILE *fp = fopen("/dev/tty", "ab");
    if (fp == NULL) {
        fprintf(stderr, "Can't open the file!\n");
        return -1;
    }
    fprintf(fp, "Hello world!\nThis directly goes to terminal (i.e., /dev/tty)!\n");
    fprintf(fp, "\"> /dev/null 2>&1\" can't redirect it!\n\n\n");
    fclose(fp);

    fp = fopen("/dev/stdout", "ab");
    if (fp == NULL) {
        fprintf(stderr, "Can't open the file!\n");
        return -1;
    }
    fprintf(fp, "On the contrary, this goes to /dev/stdout, so you can redirect it!\n\n\n");
    fclose(fp);

    uint8_t buf[4];
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Can't open the file!\n");
        return -1;
    }
    if (read(fd, buf, 8) < 0) {
        fprintf(stderr, "Can't read from the file!\n");
        close(fd);
        return -1;
    }
    close(fd);
    fprintf(stdout, "Now trying to read a 32-bit pseudo-random integer from /dev/urandom:\n");
    for(int i = 0; i < sizeof(buf)/sizeof(buf[0]); ++i)
        printf("%02X", buf[i]);
    printf("\n");
    return 0;
}