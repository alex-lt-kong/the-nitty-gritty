#include "common.h"

#include <fcntl.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>


int main(const int, char *argv[]) {

    sem_t *semptr = sem_open(SEM_NAME, O_RDWR);
    struct timespec ts;
    if (semptr == SEM_FAILED) {
        perror("sem_open()");
        goto err_sem_open;
    }

    printf("sem_wait()'ing\n");
    if (sem_wait(semptr) < 0) {
        perror("sem_wait()");
        goto err_sem_wait;
    }
    printf("CRITICAL SECTION entered\n");
    printf("Press Enter to quit the critical section\n");
    while (getchar() != '\n') {
    };

    timespec_get(&ts, TIME_UTC);
    if (sem_post(semptr) < 0) {
        perror("sem_post()");
        goto err_sem_post;
    }
    printf("sem_post()'ed (i.e., left CRITICAL SECTION) at: %ld.%09ld\n",
           ts.tv_sec, ts.tv_nsec);


err_sem_post:
err_sem_wait:
    sem_close(semptr);
    // sem_close() releases the semaphore from the program, sem_unlink()
    // releases the semaphore for the entire OS
    // sem_unlink(SEM_NAME); // on-demand program does not own the semaphore
err_sem_open:
    printf("%s exited gracefully\n", argv[0]);
    return 0;
}
