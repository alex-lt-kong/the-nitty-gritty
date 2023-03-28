#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

volatile sig_atomic_t e_flag = 0;

static void signal_handler(int signo) {
    char msg[] = "Signal [ ] caught\n";
    msg[8] = '0' + signo;
    write(STDIN_FILENO, msg, strlen(msg));
    e_flag = 1;
}

int main() {
    if (signal(SIGINT, signal_handler) == SIG_ERR) {
        perror("signal()");
        return EXIT_FAILURE;
    }
    printf("A signal handler is installed, "
        "press Ctrl+C to exit the event loop gracefully.\n");
    while (e_flag == 0) {
        sleep(2);
        printf("Event loop is running now...\n");
    }
    return EXIT_SUCCESS;
}