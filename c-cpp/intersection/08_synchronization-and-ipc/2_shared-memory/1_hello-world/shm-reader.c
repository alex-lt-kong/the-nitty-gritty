#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
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
    int fd = shm_open(SHM_NAME, O_RDONLY, SHM_PERMS);
    // O_RDWR: open an existing shm for rw
    // O_RDONLY: readonly
        if (fd < 0) {
        perror("shm_open()");
        rc = -1;
        goto err_shm_open;
    }

    /* get a pointer to memory */
    char* memptr = (char*)mmap(
        NULL,        /* addr set to NULL so that the kernel chooses the address  */
        SHM_SIZE,    /* bytes since addr */
        PROT_READ,   /* On writer side, we have PROT_READ | PROT_WRITE */
        MAP_SHARED,  /* mapping visible to other processes */
        fd,          /* file descriptor */
        0            /* offset since fd: 0 to start from the beginning */
    );
    if (memptr == MAP_FAILED) {
        rc = -1;
        perror("mmap()");
        goto err_mmap;
    }
    printf("shared mem address: %p [0..%d]\n", memptr, SHM_SIZE - 1);
    write(STDOUT_FILENO, memptr, SHM_SIZE); /* one byte at a time */

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
