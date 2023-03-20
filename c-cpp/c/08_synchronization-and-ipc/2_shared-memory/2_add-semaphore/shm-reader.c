#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>
#include <time.h>

#include "common.h"


int main() {

    int rc = 0;
    int fd = shm_open(SHM_NAME, O_RDWR, PERMS);  /* empty to begin */
    if (fd < 0) {
        rc = -1;
        perror("shm_open()");
        goto err_shm_open;
    }

    char* memptr = (char*)mmap(NULL, SHM_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if (memptr == MAP_FAILED) {
        perror("mmap()");
        rc = -1;
        goto err_mmap;
    }

    /* create a semaphore for mutual exclusion */
    sem_t* semptr = sem_open(SEM_NAME, O_RDWR);
    if (semptr == SEM_FAILED) {
        perror("sem_open()");
        rc = -1;
        goto err_sem_open;
    }

    printf("sem_wait()'ing\n");
    if (sem_wait(semptr) < 0) { perror("sem_wait()"); }
    
    timespec_get(&ts, TIME_UTC);
    printf("sem_wait()'ed at %ld.%09ld\n", ts.tv_sec, ts.tv_nsec);
    write(STDOUT_FILENO, memptr, 256);
    printf("\n......%d bytes in shared memory truncated......\n",
        SHM_SIZE - 256 * 2);
    write(STDOUT_FILENO, memptr + SHM_SIZE - 256, 256); /* one byte at a time */
    printf("\n");
    printf("The shared memory now can only be accessed by the reader, "
        "press any key to release the lock (i.e., sem_post())\n");
    getchar();

    if (sem_post(semptr) != 0) { perror("sem_post()"); }

    /* clean up */
    if (sem_close(semptr) != 0) { perror("sem_close()"); }
err_sem_open:
    if (munmap(memptr, SHM_SIZE) != 0) { perror("munmap()"); }
err_mmap:
    /* On the reader side, we need to close() ONLY */
    if (close(fd) != 0) { perror("close()"); }  
err_shm_open:
    return rc;
}
