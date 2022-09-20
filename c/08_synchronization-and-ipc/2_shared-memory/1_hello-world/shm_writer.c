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


void report_and_exit(const char* msg) {
  perror(msg);
  exit(-1);
}

int main() {
  
  int fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, SHM_PERMS);
  if (fd < 0) report_and_exit("shm_open()");

  ftruncate(fd, SHM_SIZE);
  /* Despite the confusing name, it essentially means something similar to malloc() */

  /* get a pointer to memory */
  void* memptr = mmap(
    NULL,                   /* addr set to NULL so that the kernel chooses the address  */
    SHM_SIZE,               /* bytes since addr */
    PROT_READ | PROT_WRITE, /* access protections */
    MAP_SHARED,             /* mapping visible to other processes */
    fd,                     /* file descriptor */
    0                       /* offset since fd: 0 to start from the beginning */
  );
  if ((void*) -1  == memptr) report_and_exit("mmap()");

  printf("shared mem address: %p [0..%d]\n", memptr, SHM_SIZE - 1);
  memset(memptr, 'Z', SHM_SIZE);
  printf("Shared mem written, press Enter to exit\n");
  getchar();
  
  /* clean up */
  munmap(memptr, SHM_SIZE); /* unmap the storage */
  close(fd);
  shm_unlink(SHM_NAME); /* unlink from the backing file */
  return 0;
}