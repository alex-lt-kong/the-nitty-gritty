#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "common.h"

void* func_that_takes_params(void* tpl) {
    sleep(1);
    size_t* ret = malloc(sizeof(size_t));
    if (ret != NULL) {
        *ret = strlen(((struct ThreadPayload*)tpl)->message);
    } else {
        perror("malloc()");
    }
    printf("Hello world from func_that_takes_params()!\n");
    printf("Message delivered by caller: %s\n", ((struct ThreadPayload*)tpl)->message);
    return (void*)ret;
}

int main(void) {

    struct ThreadPayload tpl;
    tpl.thread_id = 0;
    tpl.message = "This is a test message";
    pthread_t th;
    if (pthread_create(&th, NULL, func_that_takes_params, &tpl) != 0) {
        perror("pthread_create()");
        return 1;
    }
    size_t* ret;
    /* The  pthread_join() function waits for the thread specified by thread
    to terminate.  If that thread has already terminated, then pthread_join()
    returns immediately.  The thread specified by thread must be joinable.*/
    if (pthread_join(th, (void**)&ret) != 0) {
        perror("pthread_join()");
        return 1;
    }
    if (ret != NULL ) {
        printf("ret: %u\n", *ret);
    } else {
        printf("ret is NULL\n");
    }
    free(ret);
    return 0;
}