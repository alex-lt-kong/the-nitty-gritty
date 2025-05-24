#include "pthread-mutex-in-shm-common.h"

#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

int main(const int, char *argv[]) {
    char buffer[SHM_SIZE];
    int rc = 0;
    struct timespec ts;
    const int fd = shm_open(SHM_NAME, O_RDWR, SHM_PERMS);
    // O_RDWR: open an existing shm for rw
    // O_RDONLY: readonly
    if (fd < 0) {
        perror("shm_open()");
        rc = -1;
        goto err_shm_open;
    }

    /* get a pointer to memory */
    char *memptr = mmap(
            NULL, /* addr set to NULL so that the kernel chooses the address  */
            SHM_SIZE, /* bytes since addr */
            PROT_READ | PROT_WRITE,
            MAP_SHARED, /* mapping visible to other processes */
            fd, /* file descriptor */
            0 /* offset since fd: 0 to start from the beginning */
    );
    if (memptr == MAP_FAILED) {
        rc = -1;
        perror("mmap()");
        goto err_mmap;
    }
    printf("shared mem address: %p [0..%d]\n", memptr, SHM_SIZE - 1);
    pthread_mutex_t *shared_mutex = (pthread_mutex_t *) memptr;
    if ((rc = pthread_mutex_lock(shared_mutex)) != 0) {
        fprintf(stderr, "pthread_mutex_lock() failed, rc: %d\n", rc);
    }
    write(STDOUT_FILENO, memptr + sizeof(pthread_mutex_t), SHM_SIZE);
    timespec_get(&ts, TIME_UTC);
    snprintf(buffer, SHM_SIZE - sizeof(pthread_mutex_t) - 1,
             "Shared memory written by %s at %ld.%09ld\n", argv[0], ts.tv_sec,
             ts.tv_nsec);
    printf("Written the below to shm:\n%s\n", buffer);
    strncpy(memptr + sizeof(pthread_mutex_t), buffer,
            SHM_SIZE - sizeof(pthread_mutex_t) - 1);
    printf("Press Enter to quit the CRITICAL SECTION\n");
    while (getchar() != '\n') {
    };
    if ((rc = pthread_mutex_unlock(shared_mutex)) != 0) {
        fprintf(stderr, "pthread_mutex_unlock() failed, rc: %d\n", rc);
    }

    /* cleanup */
    if (munmap(memptr, SHM_SIZE) != 0) {
        perror("munmap()");
    }
err_mmap:
    /* On the reader side, we ONLY want to close() the shared memory object.
    If we also shm_unlink() it, the next shm-reader won't be able to re-use
    the same shared memory object*/
    if (close(fd) != 0) {
        perror("close()");
    }
err_shm_open:
    return rc;
}
