#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>
#include <time.h>
#include <time.h>

#include "common.h"


int main() {
    char shm_boundary[128];
    int rc = 0;
    int fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, PERMS);
    if (fd < 0) {
        rc = -1;
        perror("shm_open()");
        goto err_shm_open;
    }

    if (ftruncate(fd, SHM_SIZE) != 0) {
        perror("ftruncate()");
        rc = -1;
        goto err_ftruncate;
    }

    char* memptr = (char*)mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (memptr == MAP_FAILED) {
        perror("mmap()");
        rc = -1;
        goto err_mmap;
    }

    printf("shared mem address: %s [0..%d]\n", memptr, SHM_SIZE - 1);
    // On Linux, one can check the status of semaphore items by ls -alh /dev/shm
    sem_t* semptr = sem_open(SEM_NAME,
                             O_CREAT,            /* create the semaphore */
                             PERMS,              /* perms */
                             SEM_INITIAL_VALUE); /* initial value */
                
    if (semptr == SEM_FAILED) {
        perror("sem_open()");
        rc = -1;
        goto err_sem_open;
    }

    if (sem_wait(semptr) < 0) {
        perror("sem_wait()");
        rc = -1;
    };
    printf("sem_wait()'ed (i.e., semaphore set), "
        "press any key to memset() then sem_post()\n");
    getchar();

    timespec_get(&ts, TIME_UTC);
    sprintf(shm_boundary,
        "\n========== Shared memory buffer BEGIN at %ld.%09ld ==========\n",
        ts.tv_sec, ts.tv_nsec);
    strcpy(memptr, shm_boundary);
    memset(memptr + strlen(shm_boundary) + 1, 'Y',
        SHM_SIZE - (strlen(shm_boundary) + 1));
    timespec_get(&ts, TIME_UTC);
    sprintf(shm_boundary,
        "\n========== Shared memory buffer END at %ld.%09ld ==========\n",
        ts.tv_sec, ts.tv_nsec);
    strcpy(memptr + SHM_SIZE - (strlen(shm_boundary) + 1), shm_boundary);
    if (sem_post(semptr) != 0) {
        perror("sem_post()");
        rc = -1;
    }

    /* clean up */    
    if (sem_unlink(SEM_NAME) != 0) { perror("sem_unlink()"); }
    if (sem_close(semptr) != 0) { perror("sem_close()"); }
err_sem_open:
    if (munmap(memptr, SHM_SIZE) != 0) { perror("munmap()"); }
err_mmap:
err_ftruncate:
    /* On the writer side, we need to both shm_unlink() and close() */
    if (shm_unlink(SHM_NAME) != 0) { perror("shm_unlink()"); }
    if (close(fd) != 0) { perror("close()"); }  
err_shm_open:
    return rc;
}
