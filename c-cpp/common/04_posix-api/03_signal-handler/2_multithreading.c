#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>

volatile sig_atomic_t e_flag = 0;

static void signal_handler(int signo) {
    char msg[] = "Signal [ ] caught\n";
    msg[8] = '0' + signo;
    write(STDIN_FILENO, msg, strlen(msg));
    e_flag = 1;
}

void* event_loop(void* param) {
    size_t tid = *((size_t*)param);
    size_t iter_count = 0;
    while (e_flag == 0) {
        ++ iter_count;
        printf("Th %lu: Event loop is running now, iterated %lu times ...\n",
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

int main(void) {
    if (signal(SIGINT, signal_handler) == SIG_ERR) {
        perror("signal()");
        return EXIT_FAILURE;
    }
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
                exit(EXIT_FAILURE);
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
                printf("Th %lu exited, iterated: %lu times\n", i, *ret);
                free(ret);
            } else {
                printf("Th %lu exited, but retval is not set as expected\n", i);
            }
        }
    }
    return EXIT_SUCCESS;
}
