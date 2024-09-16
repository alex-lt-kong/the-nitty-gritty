#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


void* naive_func() {
    sleep(1);
    int* ret = (int*)malloc(sizeof(int));
    if (ret != NULL) {
        *ret = 1;
    } else {
        perror("malloc()");
    }
    printf("Hello world from naive_func()!\n");
    return (void*)ret;
}


void* naive_func_self_detaching() {
    if (pthread_detach(pthread_self()) != 0) {
        perror("pthread_detach()");
    }
    sleep(1);
    printf("Hello world from naive_func_self_detaching()!\n");
    /* Seems there is not straightforward way to catch the return value from
    a detached function, so we have to return NULL; otherwise we may leak
    memory.
    pthread_exit() always succeeds. */
    pthread_exit(NULL);
}


int main(void) {


    /* Both pthread_join() and pthread_detach() can be called upon an exited
    thread.*/
    pthread_t tid;
    if (pthread_create(&tid, NULL, naive_func, NULL) != 0) {
        perror("pthread_create()");
        return 1;
    }
    int* ret;
    /* The  pthread_join() function waits for the thread specified by thread
    to terminate.  If that thread has already terminated, then pthread_join()
    returns immediately.  The thread specified by thread must be joinable.*/
    if (pthread_join(tid, (void**)&ret) != 0) {
        perror("pthread_join()");
        return 1;
    }
    /* We must call either pthread_join() or pthread_detach() exactly once,
    no zero times, not both, not calling one twice. Calling both or calling
    one of them twice invokes undefined behaviors. */
    if (ret != NULL) {
        printf("ret: %d\n", *ret);
        free(ret);
    } else {
        printf("naive_func() returns NULL!\n");
    }


    if (pthread_create(&tid, NULL, naive_func, NULL) != 0) {
        perror("pthread_create()");
        return 1;
    }
    /* One int worth of memory is leaked, as we don't have the chance to
    free(ret); */
    if (pthread_detach(tid) != 0) {
        perror("pthread_detach()");
        return 1;
    }
    if (pthread_create(&tid, NULL, naive_func_self_detaching, NULL) != 0) {
        perror("pthread_create()");
        return 1;
    }


    if (pthread_create(&tid, NULL, naive_func, NULL) != 0) {
        perror("pthread_create()");
        return 1;
    }
    if (pthread_join(tid, (void**)&ret) != 0) {
        perror("pthread_join()");
        return 1;
    }
    if (ret != NULL) {
        printf("ret: %d\n", *ret);
        free(ret);
    } else {
        printf("naive_func() returns NULL!\n");
    }

    /* This Hello world! won't be printf()ed as the main thread returns
    before it can happen*/
    if (pthread_create(&tid, NULL, naive_func, NULL) != 0) {
        perror("pthread_create()");
        return 1;
    }
    /* One int worth of memory is leaked, as we don't have the chance to
    free(ret); */
    if (pthread_detach(tid) != 0) {
        perror("pthread_detach()");
        return 1;
    }
    return 0;
}
