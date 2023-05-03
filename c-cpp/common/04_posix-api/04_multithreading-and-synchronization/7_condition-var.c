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
pthread_cond_t my_cv;


void* writing_func(void* tpl) {
    struct ThreadPayload* args = (struct ThreadPayload*)tpl;
    
    while (!should_stop) {
        // usleep(1000 * 1000 - args->thread_id);
        char* buf = (char*)malloc(sizeof(char) * 512);
        if (buf != NULL) {
            if (pthread_mutex_lock(&my_mutex) != 0) {
                perror("pthread_mutex_lock()");
                continue;
            }
            sprintf(buf, "[%lu] Message from caller: %s\n",
                args->thread_id, args->message);
            g_queue_push_tail(q, buf);
            /* 
            * pthread_cond_signal() restarts  one  of  the threads that are
            waiting on the condition variable cond. If no threads are waiting
            on cond, nothing happens. If several threads are waiting on cond,
            exactly one is restarted, but it is not specified which.
            
            * Note that pthread_cond_signal() sends a signal only,
            we still need to pthread_mutex_unlock() so that pthread_cond_wait()
            can return by locking the mutex as suggests in this SO post:
            https://stackoverflow.com/questions/4544234/calling-pthread-cond-signal-without-locking-mutex
            */
            
            pthread_cond_signal(&my_cv);
            if (pthread_mutex_unlock(&my_mutex) != 0) {
                perror("pthread_mutex_unlock() error, "
                    "writing_func() exited prematurely");
                return NULL;
            }
            
        } else {
            perror("malloc() error, writing_func() exited prematurely");
            return NULL;
        }
    }
    printf("[%lu] writing_func() exited gracefully\n", args->thread_id);
    return NULL;
}

void* reading_func() {

    while (!should_stop) {
        usleep(1);
        if (pthread_mutex_lock(&my_mutex) != 0) {
            perror("pthread_mutex_lock()");
            continue;
        }
        
        /* pthread_cond_wait() unlocks my_mutex before it blocks this
        thread. */
        int rc = pthread_cond_wait(&my_cv, &my_mutex);
        /* Before returning to the calling thread, pthread_cond_wait()
        re-acquires mutex (as per pthread_lock_mutex). This also implies that
        we still need to pthread_mutex_unlock() so that pthread_cond_wait()
        can return by successfully locking the mutex. */
        if (rc != 0) {
            perror("pthread_cond_wait()");
            continue;
        }
        
        if (!g_queue_is_empty(q)) {
            printf("=== reading START ===\n");
            while (!g_queue_is_empty(q)) {
                /* We can further optimize by removing printf() and free()
                from the critical section. */
                char* buf = (char*)g_queue_pop_head(q);
                if (buf != NULL) {
                    printf("%s", buf);
                    free(buf);
                } else {
                    printf("Head of the internal queue is NULL!\n");
                }
            }
            printf("=== reading END ===\n");
        } else {
            printf("g_queue_is_empty(q) is true but unlocked\n");
        }
        if (pthread_mutex_unlock(&my_mutex) != 0) {
            perror("pthread_mutex_unlock()");
            return NULL;
        }
        //printf("reading_func() unlocked\n");
    }
    printf("reading_func() exited gracefully\n");
    return NULL;
}


int main(void) {
    install_signal_handler();
    int retval = 0;
    q = g_queue_new();
    /* g_queue_new()'s doc does not explicitly say it can return NULL. After
    following its call stack, it seems that malloc() is used internally,
    so NULL is possible. */
    if (q == NULL) {
        perror("g_queue_new()");
        retval = 1;
        goto new_queue_err;
    }
    should_stop = false;
    if (pthread_mutex_init(&my_mutex, NULL) != 0) {
        perror("pthread_mutex_init()");
        retval = 1;
        goto mutex_init_err;
    }
    /* Depending on which platform is being used. On Debian,
    pthread_cond_init() never returns abn error code */
    if (pthread_cond_init(&my_cv, NULL) != 0) {
        perror("pthread_cond_init()");
        retval = 1;
        goto cv_init_err;
    }

    size_t thread_count = sizeof(thread_payloads)/sizeof(thread_payloads[0]);
    {
        pthread_t ths[thread_count];
        pthread_t read_th;
        size_t good_threads;
        for (good_threads = 0; good_threads < thread_count; ++good_threads) {
            if (pthread_create(&ths[good_threads], NULL, writing_func,
                &thread_payloads[good_threads]) != 0) {
                perror("pthread_create()");
                --good_threads;
                break;
            }
        }
        printf("%lu writing threads started\n", good_threads);
        if (pthread_create(&read_th, NULL, reading_func, NULL) != 0) {
            perror("pthread_create()");
        }
        pthread_join(read_th, NULL);
        for (size_t i = 0; i < good_threads; ++i) {
            /* The  pthread_join() function waits for the thread specified
            by thread to terminate.  If that thread has already terminated,
            then pthread_join() returns immediately.  The thread specified
            by thread must be joinable.*/
            if (pthread_join(ths[i], NULL) != 0) {
                perror("pthread_join()");
                continue;
            }
        }
    }

    while (!g_queue_is_empty(q)) {
        char* buf = (char*)g_queue_pop_head(q);
        g_queue_remove(q, NULL);
        if (buf != NULL) {
            free(buf);
        }
    }

cv_init_err:
    if (pthread_mutex_destroy(&my_mutex) != 0) {
        // But there is nothing else we can do on this.
        perror("pthread_mutex_destroy()");
    }
mutex_init_err:
    g_queue_free(q);
new_queue_err:
    return retval;
}
