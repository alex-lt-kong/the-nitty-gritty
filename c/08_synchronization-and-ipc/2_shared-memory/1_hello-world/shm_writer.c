/** Compilation: gcc -o memwriter memwriter.c -lrt -lpthread **/
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
  int AccessPerms = 0644;
  char SemaphoreName[] = "MySemaphore";
  char MemContents[1023];
  memset(MemContents, 'C', 1023);
  MemContents[1023] = '\0';
  int fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, SEM_PERMS);
  if (fd < 0) report_and_exit("shm_open()");

  ftruncate(fd, SHM_SIZE);
  /* Despite the confusing name, it essentially means something similar to malloc() */

  void* memptr = mmap(NULL,       /* let system pick where to put segment */
                      SHM_SIZE,   /* how many bytes */
                      PROT_READ | PROT_WRITE, /* access protections */
                      MAP_SHARED, /* mapping visible to other processes */
                      fd,         /* file descriptor */
                      0);         /* offset: start at 1st byte */
  if ((void*) -1  == memptr) report_and_exit("mmap()");

  fprintf(stderr, "shared mem address: %p [0..%d]\n", memptr, SHM_SIZE - 1);

  sem_t* semptr = sem_open(SEM_NAME,
                           O_CREAT,       /* create the semaphore */
                           SEM_PERMS,   /* protection perms */
                           SEM_INITIAL_VALUE);            /* initial value */
  if (semptr == (void*) -1) report_and_exit("sem_open()");
  if (sem_wait(semptr) < 0) report_and_exit("sem_wait()");
  printf("sleep()'ing for 10 sec, emulating data fetching\n");
  sleep(10);
  
  strcpy(memptr, MemContents); /* copy some ASCII bytes to the segment */
 
  /* increment the semaphore so that memreader can read */
  if (sem_post(semptr) < 0) report_and_exit("sem_post()");
  printf("sem_post()'ed\n");
  sleep(20); /* give reader a chance */

  /* clean up */
  munmap(memptr, SHM_SIZE); /* unmap the storage */
  close(fd);
  sem_close(semptr);
  shm_unlink(SHM_NAME); /* unlink from the backing file */
  return 0;
}