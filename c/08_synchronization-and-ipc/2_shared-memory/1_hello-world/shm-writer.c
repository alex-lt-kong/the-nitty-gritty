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
    /* On Linux, one can check the status of shared memory items by
    ls -alh /dev/shm
    One may also notice that PERMS are not fully effective, we need to call
    umask() to make it work:
    https://stackoverflow.com/questions/51068208/shm-open-not-setting-group-write-access
    */
    int fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, SHM_PERMS);
    /* O_RDWR: open an existing shm for rw
       O_CREAT: create a new shm if it doesn't exist
       We should specify O_CREAT only on the writer side, or reader may
       inadvertently create a brand new shm object.
    */
    if (fd < 0) {
        perror("shm_open()");
        rc = -1;
        goto err_shm_open;
    }
    
    /* Despite the confusing name, it essentially means something
    similar to malloc() */
    if (ftruncate(fd, SHM_SIZE) != 0) {
        perror("ftruncate()");
        rc = -1;
        goto err_ftruncate;
    }

    /* get a pointer to memory */
    char* memptr = (char*)mmap(
        NULL,                   /* addr set to NULL so that the kernel chooses the address  */
        SHM_SIZE,               /* bytes since addr */
        PROT_READ | PROT_WRITE, /* access protections */
        MAP_SHARED,             /* mapping visible to other processes */
        fd,                     /* file descriptor */
        0                       /* offset since fd: 0 to start from the beginning */
    );
    if (memptr == MAP_FAILED) {
        perror("mmap()");
        rc = -1;
        goto err_mmap;
    }

    printf("shared mem address: %p [0..%d]\n", memptr, SHM_SIZE - 1);
    memset(memptr, 88, SHM_SIZE);
    printf("Shared mem written, press Enter to exit\n");
    getchar();
    
    /* clean up */
    /* unmap the storage */   
    /* This is no C++ and no RAII, so nothing guarantees that we can't have
     error during finalization. */
    if (munmap(memptr, SHM_SIZE) != 0) {
        perror("munmap()");
    }
err_ftruncate:
err_mmap:
    /* On the writer side, we want to both shm_unlink() and close() the shared
    memory object. By shm_unlink()ing the shm object, we only remove the name
    of a referenced shm object. It is like a pointer goes out of scope--we
    can no longer use the pointer to refer to the shm object, but the object
    on heap won't be automatically gone. To get ride of the object,
    we need to close() it, just like we need to free() a pointer 
    Note with a pointer, we must free() it once, or we are doomed. For
    close(), each process that shm_open()'s it should close() once, the 
    underlying resources will be released when the last process close()'s it.*/
    if (shm_unlink(SHM_NAME) != 0) {
        /* We check all possible errors, but actually there is nothing we
        can do expect sending the error to stderr.*/
        perror("shm_unlink()");
    }
    if (close(fd) != 0) {
        perror("close()");
    }
err_shm_open:
    return rc;
}