#include <atomic>
#include <signal.h>

extern std::atomic<bool> should_stop;

void stop_all_threads() {
    should_stop = true;
}

static void signal_handler(int signum) {
    char msg[] = "Signal [  ] caught\n";
    msg[8] = '0' + signum / 10;
    msg[9] = '0' + signum % 10;
    size_t len = sizeof(msg) - 1;
    ssize_t written = 0;
    while (written < len) {
        ssize_t ret = write(STDOUT_FILENO, msg + written, len - written);
        if (ret == -1) {
            // perror() is not reentrant thus can't be used here
            break;
        }
        written += ret;
    }
    stop_all_threads();
}

void install_signal_handler() {
    struct sigaction act;
    if (sigemptyset(&act.sa_mask) == -1) {
        perror("sigemptyset()");
        abort();
    }
    act.sa_handler = signal_handler;  
    act.sa_flags = 0;
    if (sigaction(SIGINT,  &act, 0) + sigaction(SIGABRT, &act, 0) +
        sigaction(SIGQUIT, &act, 0) + sigaction(SIGTERM, &act, 0) < 0) {
        perror("sigaction()");
        abort();
    }
}
