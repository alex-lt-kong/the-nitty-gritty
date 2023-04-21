#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "common.h"

// Without synchronization, this shared memory space will be overwritten
// concurrently, causing unexpected (yet probably still well-defined) behaviors.
static char shared_string[512];


void* func_that_takes_params(void* tpl) {
    struct ThreadPayload* args = (struct ThreadPayload*)tpl;
    while (!should_stop) {
        // Let's add a bit of perturbation, but not too much
        usleep(1000 * 1000 - args->thread_id);
        sprintf(shared_string, "[%d] Message from caller: %s\n",
            args->thread_id, args->message);
        write(STDOUT_FILENO, shared_string, strlen(shared_string));
    }
    size_t* ret = malloc(sizeof(size_t));
    if (ret != NULL) {
        *ret = strlen(args->message);
    } else {
        perror("malloc()");
    }
    return (void*)ret;
}

int main(void) {
    should_stop = false;
    size_t thread_count = sizeof(thread_payloads)/sizeof(thread_payloads[0]);
    install_signal_handler();

    pthread_t ths[thread_count];
    size_t started_threads;
    for (started_threads = 0; started_threads < thread_count; ++started_threads) {
        if (pthread_create(&ths[started_threads], NULL, func_that_takes_params,
            &thread_payloads[started_threads]) != 0) {
            perror("pthread_create()");
            --started_threads;
            break;
        }
    }
    
    for (int i = 0; i < started_threads; ++i) {
        size_t* ret = NULL;
        /* The  pthread_join() function waits for the thread specified by thread
        to terminate.  If that thread has already terminated, then pthread_join()
        returns immediately.  The thread specified by thread must be joinable.*/
        if (pthread_join(ths[i], (void**)&ret) != 0) {
            perror("pthread_join()");
            continue;
        }
        if (ret != NULL) {
            printf("ret: %u\n", *ret);
        } else {
            printf("ret is NULL\n");
        }
        free(ret);
    }
    return 0;
}
