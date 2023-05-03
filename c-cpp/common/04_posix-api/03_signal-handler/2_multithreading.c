#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>

volatile sig_atomic_t e_flag = 0;

static void signal_handler(int signum) {
    char msg[] = "Signal [  ] caught\n";
    msg[8] = '0' + (char)(signum / 10);
    msg[9] = '0' + (char)(signum % 10);
    ssize_t len = sizeof(msg) - 1;
    ssize_t written = 0;
    while (written < len) {
        ssize_t ret = write(STDOUT_FILENO, msg + written, len - written);
        if (ret == -1) {
            // perror() is not reentrant thus can't be used here
            break;
        }
        written += ret;
    }
    e_flag = 1;
}

void* event_loop(void* param) {
    size_t tid = *((size_t*)param);
    size_t iter_count = 0;
    while (e_flag == 0) {
        ++ iter_count;
        printf("Th %zu: Event loop is running now, iterated %zu times ...\n",
            tid, iter_count);
        for (size_t i = 0; i < tid + 1 && e_flag == 0; ++ i) {
            sleep(1);
        }
    }
    size_t* ret = (size_t*)malloc(sizeof(size_t*));
    if (ret != NULL) {
        *ret = iter_count;
    } else {
        perror("malloc()");
    }
    return ret;
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
    /*
    * SA_RESETHAND means we want our signal_handler() to intercept the signal
    once. If a signal is sent twice, the default signal handler will be used
    again. `man sigaction` describes more possible sa_flags.
    * In this particular case, we should not enable SA_RESETHAND, mainly
    due to the issue that if a child process is kill, multiple SIGPIPE will
    be invoked consecutively, breaking the program.
    * Without setting SA_RESETHAND, catching SIGSEGV is usually a bad idea.
    The issue is, if an instruction results in segfault, SIGSEGV handler is
    called, then the very same instruction will be repeated, triggering
    segfault again. */
    // act.sa_flags = SA_RESETHAND;    
    act.sa_flags = 0;
    if (sigaction(SIGINT,  &act, 0) + sigaction(SIGABRT, &act, 0) +
        sigaction(SIGQUIT, &act, 0) + sigaction(SIGTERM, &act, 0) +
        sigaction(SIGPIPE, &act, 0) + sigaction(SIGCHLD, &act, 0) +
        sigaction(SIGTRAP, &act, 0) < 0) {
        
        /* Could miss some error if more than one sigaction() fails. However,
        given that the program will quit if one sigaction() fails, this
        is not considered an issue */
        perror("sigaction()");
        abort();
    }
}

int main(void) {

    install_signal_handler();
    printf("A signal handler is installed, "
        "press Ctrl+C to exit event loop threads gracefully.\n");

    pthread_t threads[8];
    size_t running_thread_count = 0;
    /* It is not guaranteed that pthread_t is an int type, so we'd better
    create our own thread_id*/
    size_t tids[sizeof(threads) / sizeof(threads[0])];
    for (size_t i = 0; i < sizeof(threads) / sizeof(threads[0]); ++i) {
        tids[i] = i;
        int err_no;
        // pthread_create() doesn't set errno,
        // we need to catch its retval manually
        if ((err_no = pthread_create(
            &threads[i], NULL, event_loop, (void *)&tids[i])) != 0) {
            fprintf(stderr, "pthread_create() failed: %d(%s), "
                "the program exits now", err_no, strerror(err_no));
            if ((err_no = raise(SIGINT)) != 0) {
                fprintf(stderr, "even raise() failed: %d(%s), "
                    "the program exits UNgracefully",
                    err_no, strerror(err_no));
                /* Refer to man abort on why abort() could be better than
                exit(EXIT_FAILURE) */
                abort();
            }
            break;
        }
        ++running_thread_count;
    }

    for (size_t i = 0; i < running_thread_count; ++i) {
        size_t* ret;
        int err_no;
        // pthread_join() doesn't set errno,
        // we need to catch its retval manually
        if ((err_no = pthread_join(threads[i], (void**)&ret)) != 0) {
            fprintf(stderr, "pthread_join() failed: %d(%s), "
                "but there is nothing much we can do",
                err_no, strerror(err_no));
        } else {
            if (ret != NULL) {
                printf("Th %zu exited, iterated: %zu times\n", i, *ret);
                free(ret);
            } else {
                printf("Th %zu exited, but retval is not set as expected\n", i);
            }
        }
    }
    return EXIT_SUCCESS;
}
