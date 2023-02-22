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
  int fd = shm_open(SHM_NAME, O_RDWR, PERMS);  /* empty to begin */
  if (fd < 0) report_and_exit("shm_open()");

  void* memptr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if ((void*) -1 == memptr) report_and_exit("mmap()");

  /* create a semaphore for mutual exclusion */
  sem_t* semptr = sem_open(SEM_NAME, O_RDWR);
  if (semptr == (void*) -1) report_and_exit("sem_open()");

  printf("sem_wait()'ing\n");
  if (sem_wait(semptr) < 0) report_and_exit("sem_wait()");
  
  timespec_get(&ts, TIME_UTC);
  printf("sem_wait()'ed at %ld.%09ld\n", ts.tv_sec, ts.tv_nsec);
  write(STDOUT_FILENO, memptr, 256);
  printf("\n......%d bytes in shared memory truncated......\n", SHM_SIZE - 256 * 2);
  write(STDOUT_FILENO, memptr + SHM_SIZE - 256, 256); /* one byte at a time */
  printf("\n");
  sem_post(semptr);


  /* cleanup */
  munmap(memptr, SHM_SIZE);
  close(fd);
  sem_close(semptr);
  unlink(SEM_NAME);
  return 0;
}