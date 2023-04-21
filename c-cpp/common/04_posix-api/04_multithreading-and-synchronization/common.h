#include <signal.h>
#include <stdbool.h>

volatile sig_atomic_t should_stop;

struct ThreadPayload {
    size_t thread_id;
    char* message;
};

static struct ThreadPayload thread_payloads[] = {
    {
        .thread_id = 0,
        .message = "This is a test message"
    }, {
        .thread_id = 1,
        .message = "This is also a test message"
    }, {
        .thread_id = 2,
        .message = "This is yet another a test message"
    }, {
        .thread_id = 3,
        .message = "Hello world!!"
    }, {
        .thread_id = 4,
        .message = "foobar"
    }, {
        .thread_id = 5,
        .message = "0xDEADBEEF"
    }, {
        .thread_id = 6,
        .message = "The quick brown fox jumps over the lazy dog"
    }, {
        .thread_id = 7,
        .message = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    }
};

static void signal_handler(int signum) {
    char msg[] = "Signal [  ] caught\n";
    msg[8] = '0' + signum / 10;
    msg[9] = '0' + signum % 10;
    size_t len = sizeof(msg) - 1;
    size_t written = 0;
    while (written < len) {
        ssize_t ret = write(STDOUT_FILENO, msg + written, len - written);
        if (ret == -1) {
            perror("write()");
            break;
        }
        written += ret;
    }
    should_stop = true;
}

// This function calls abort() if fails
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