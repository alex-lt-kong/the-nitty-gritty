#include <stdio.h>

int main() {
    FILE *fp = fopen("/dev/tty", "ab");
    if (fp != NULL) {
        fprintf(fp, "Hello world!\n");
        fclose(fp);
    }
    return 0;
}