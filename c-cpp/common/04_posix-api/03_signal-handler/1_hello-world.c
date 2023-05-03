#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

volatile sig_atomic_t e_flag = 0;

static void signal_handler(int signum) {
    char msg[] = "Signal [  ] caught\n";
    msg[8] = '0' + (char)(signum / 10);
    msg[9] = '0' + (char)(signum % 10);
    write(STDIN_FILENO, msg, strlen(msg));
    e_flag = 1;
}

void install_signal_handler() {
    // This design canNOT handle more than 99 signal types
    if (_NSIG > 99) {
        fprintf(stderr, "signal_handler() can't handle more than 99 signals\n");
        abort();
    }
    struct sigaction act;
    // Initialize the signal set to empty, similar to memset(0)
    if (sigemptyset(&act.sa_mask) == -1) {
        perror("sigemptyset()");
        abort();
    }
    act.sa_handler = signal_handler;
    /* SA_RESETHAND means we want our signal_handler() to intercept the signal
    once. If a signal is sent twice, the default signal handler will be used
    again. `man sigaction` describes more possible sa_flags. */
    act.sa_flags = SA_RESETHAND;
    //act.sa_flags = 0;
    if (sigaction(SIGINT, &act, 0) == -1) {
        perror("sigaction()");
        abort();
    }
}

int main(void) {
    

    install_signal_handler();
    printf("A signal handler is installed, "
        "press Ctrl+C to exit the event loop gracefully.\n");
    while (e_flag == 0) {
        sleep(2);
        printf("Event loop is running now...\n");
    }
    return EXIT_SUCCESS;
}