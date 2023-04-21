#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "common.h"


static char shared_string[512];
// No, we should not define my_mytex as volatile.
pthread_mutex_t my_mutex;


void* func_that_takes_params(void* tpl) {
    struct ThreadPayload* args = (struct ThreadPayload*)tpl;
    while (!should_stop) {
        // Let's add a bit of perturbation, but not too much
        usleep(1000 * 1000 - args->thread_id);
        if (pthread_mutex_lock(&my_mutex) != 0) {
            perror("pthread_mutex_lock()");
            continue;
        }
        sprintf(shared_string, "[%d] Message from caller: %s\n",
            args->thread_id, args->message);
        write(STDOUT_FILENO, shared_string, strlen(shared_string));
        if (pthread_mutex_unlock(&my_mutex) != 0) {
            /* This seems to be a highly unlikely scenario as POSIX specifies 
            only the below cases:
            * EINVAL：The value specified by mutex does not refer to an
            initialized mutex object. 
            * EAGAIN：The mutex could not be acquired because the maximum
            number of recursive locks for mutex has been exceeded.
            * EPERM： The current thread does not own the mutex.
            */
            perror("pthread_mutex_unlock()");
            return NULL;
        }
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
    install_signal_handler();
    should_stop = false;
    if (pthread_mutex_init(&my_mutex, NULL) != 0) {
        perror("pthread_mutex_init()");
        return 1;
    }

    size_t thread_count = sizeof(thread_payloads)/sizeof(thread_payloads[0]);

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
    printf("%d threads started\n", started_threads);
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
            // No, we don't need to cast ret to (size_t*)--it is defined as
            // size_t* in the first place!
            printf("ret: %u\n", *ret);
        } else {
            printf("ret is NULL\n");
        }
        free(ret);
    }
    if (pthread_mutex_destroy(&my_mutex) != 0) {
        // But there is nothing else we can do on this.
        perror("pthread_mutex_destroy()");
    }
    return 0;
}
