#include <pthread.h>
#include <stdio.h>

void* naive_func() {
    printf("Hello world!\n");
    return NULL;
}

int main(void) {
    pthread_t tid;
    if (pthread_create(&tid, NULL, naive_func, NULL) != 0) {
        perror("pthread_create()");
        return 1;
    }
    if (pthread_join(tid, NULL) != 0) {
        perror("pthread_join()");
        return 1;
    }
    return 0;
}
