/** Compilation: gcc -o memreader memreader.c -lrt -lpthread **/
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>

#include "common.h"

void report_and_exit(const char* msg) {
  perror(msg);
  exit(-1);
}

int main() {
  int fd = shm_open(SHM_NAME, O_RDWR, SEM_PERMS);  /* empty to begin */
  if (fd < 0) report_and_exit("shm_open()");

  /* get a pointer to memory */
  void* memptr = mmap(NULL,       /* let system pick where to put segment */
                        SHM_SIZE,   /* how many bytes */
                        PROT_READ | PROT_WRITE, /* access protections */
                        MAP_SHARED, /* mapping visible to other processes */
                        fd,         /* file descriptor */
                        0);         /* offset: start at 1st byte */
  if ((void*) -1 == memptr) report_and_exit("mmap()");

  /* create a semaphore for mutual exclusion */
  sem_t* semptr = sem_open(SEM_NAME, /* name */
                           O_CREAT,       /* create the semaphore */
                           SEM_PERMS,   /* protection perms */
                           SEM_INITIAL_VALUE);            /* initial value */
  if (semptr == (void*) -1) report_and_exit("sem_open()");

  printf("sem_wait()'ing\n");
  if (sem_wait(semptr) == 0) {
    int i;
    for (i = 0; i < 1023; i++)
      write(STDOUT_FILENO, memptr + i, 1); /* one byte at a time */
    sem_post(semptr);
  }

  /* cleanup */
  munmap(memptr, SHM_SIZE);
  close(fd);
  sem_close(semptr);
  unlink(SEM_NAME);
  return 0;
}