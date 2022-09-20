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

void report_and_exit(const char* msg) {
  perror(msg);
  exit(-1);
}

int main() {

  int fd = shm_open(SHM_NAME, O_RDWR, SHM_PERMS);
  // O_RDWR: open an existing shm for rw
  if (fd < 0) report_and_exit("shm_open()");

  /* get a pointer to memory */
  void* memptr = mmap(
    NULL,                   /* addr set to NULL so that the kernel chooses the address  */
    SHM_SIZE,               /* bytes since addr */
    PROT_READ | PROT_WRITE, /* access protections */
    MAP_SHARED,             /* mapping visible to other processes */
    fd,                     /* file descriptor */
    0                       /* offset since fd: 0 to start from the beginning */
  );
  if ((void*) -1 == memptr) report_and_exit("mmap()");

  write(STDOUT_FILENO, memptr, SHM_SIZE); /* one byte at a time */

  /* cleanup */
  munmap(memptr, SHM_SIZE);
  close(fd);
  shm_unlink(SHM_NAME);
  return 0;
}