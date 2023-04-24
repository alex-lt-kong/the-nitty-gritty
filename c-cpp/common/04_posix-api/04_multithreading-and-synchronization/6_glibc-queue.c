#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <glib.h>

#include "common.h"

static GQueue *q;

// No, we should not define my_mytex as volatile.
pthread_mutex_t my_mutex;


void* writing_func(void* tpl) {
    struct ThreadPayload* args = (struct ThreadPayload*)tpl;
    
    while (!should_stop) {
        // Let's add a bit of perturbation, but not too much
        usleep(1000 * 1000 - args->thread_id);
        if (pthread_mutex_lock(&my_mutex) != 0) {
            perror("pthread_mutex_lock()");
            continue;
        }
        char* buf = malloc(sizeof(char) * 512);
        if (buf != NULL) {
            sprintf(buf, "[%d] Message from caller: %s\n",
                args->thread_id, args->message);
            g_queue_push_tail(q, buf);
        } else {
            perror("malloc()");
        }
        if (pthread_mutex_unlock(&my_mutex) != 0) {
            perror("pthread_mutex_unlock()");
            return NULL;
        }
    }
    return NULL;
}

void* reading_func() {

    while (!should_stop) {
        sleep(1);
        if (pthread_mutex_lock(&my_mutex) != 0) {
            perror("pthread_mutex_lock()");
            continue;
        }
        while (!g_queue_is_empty(q)) {
            usleep(1);
            char* buf = g_queue_pop_head(q);
            g_queue_remove(q, NULL);
            if (buf != NULL) {
                printf("=== START ===\n%s\n=== END ===\n", buf);
                free(buf);
            } else {
                printf("1st element of the internal queue is NULL!\n");
            }
        }

        if (pthread_mutex_unlock(&my_mutex) != 0) {
            perror("pthread_mutex_unlock()");
            return NULL;
        }
    }
    return NULL;
}


int main(void) {
    install_signal_handler();
    q = g_queue_new();
    /* g_queue_new()'s doc does not explicitly say it can return NULL. After
    following its call stack, it seems that malloc() is used internally,
    so NULL is possible. */
    if (q == NULL) {
        perror("g_queue_new()");
        return 1;
    }
    should_stop = false;
    if (pthread_mutex_init(&my_mutex, NULL) != 0) {
        perror("pthread_mutex_init()");
        goto mutex_init_err;
    }

    size_t thread_count = sizeof(thread_payloads)/sizeof(thread_payloads[0]);
    {
        pthread_t ths[thread_count];
        pthread_t read_th;
        size_t started_threads;
        for (started_threads = 0; started_threads < thread_count; ++started_threads) {
            if (pthread_create(&ths[started_threads], NULL, writing_func,
                &thread_payloads[started_threads]) != 0) {
                perror("pthread_create()");
                --started_threads;
                break;
            }
        }
        printf("%d writing threads started\n", started_threads);
        if (pthread_create(&read_th, NULL, reading_func, NULL) != 0) {
            perror("pthread_create()");
        }
        pthread_join(read_th, NULL);
        for (int i = 0; i < started_threads; ++i) {
            /* The  pthread_join() function waits for the thread specified by thread
            to terminate.  If that thread has already terminated, then pthread_join()
            returns immediately.  The thread specified by thread must be joinable.*/
            if (pthread_join(ths[i], NULL) != 0) {
                perror("pthread_join()");
                continue;
            }
        }
    }
    if (pthread_mutex_destroy(&my_mutex) != 0) {
        // But there is nothing else we can do on this.
        perror("pthread_mutex_destroy()");
    }
    while (!g_queue_is_empty(q)) {
        char* buf = g_queue_pop_head(q);
        g_queue_remove(q, NULL);
        if (buf != NULL) {
            free(buf);
        }
    }

mutex_init_err:
    g_queue_free(q);
    return 0;
}
