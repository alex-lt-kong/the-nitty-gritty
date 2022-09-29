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
  struct timespec ts;
  char shm_boundary[128];
  
  int fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, PERMS);
  if (fd < 0) report_and_exit("shm_open()");

  ftruncate(fd, SHM_SIZE);  /* it means something similar to malloc() */

  void* memptr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if ((void*) -1  == memptr) report_and_exit("mmap()");

  printf("shared mem address: %p [0..%d]\n", memptr, SHM_SIZE - 1);

  sem_t* semptr = sem_open(SEM_NAME,
                           O_CREAT,            /* create the semaphore */
                           PERMS,              /* perms */
                           SEM_INITIAL_VALUE); /* initial value */
  if (semptr == (void*) -1) report_and_exit("sem_open()");

  if (sem_wait(semptr) < 0) report_and_exit("sem_wait()");
  printf("sem_wait()'ed, press any key to memset() then sem_post()\n");
  getchar();

  timespec_get(&ts, TIME_UTC);
  sprintf(shm_boundary, "\n========== Shared memory buffer BEGIN at %ld.%09ld ==========\n", ts.tv_sec, ts.tv_nsec);
  strcpy(memptr, shm_boundary);
  memset(memptr + strlen(shm_boundary) + 1, 'Y', SHM_SIZE - (strlen(shm_boundary) + 1));
  timespec_get(&ts, TIME_UTC);
  sprintf(shm_boundary, "\n========== Shared memory buffer END at %ld.%09ld ==========\n", ts.tv_sec, ts.tv_nsec);
  strcpy(memptr + SHM_SIZE - (strlen(shm_boundary) + 1), shm_boundary);
  if (sem_post(semptr) < 0) report_and_exit("sem_post()");
  /* clean up */
  munmap(memptr, SHM_SIZE); /* unmap the storage */
  close(fd);
  sem_close(semptr);
  sem_unlink(semptr);
  shm_unlink(SHM_NAME); /* unlink from the backing file */
  return 0;
}