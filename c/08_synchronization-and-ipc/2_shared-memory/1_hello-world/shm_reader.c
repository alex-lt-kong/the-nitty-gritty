/** Compilation: gcc -o memreader memreader.c -lrt -lpthread **/
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#include "common.h"

volatile sig_atomic_t done = 0;

void signal_handler(int signum) {
  char msg[] = "Signal %d received by signal_handler(), press Enter to send the signal to event loop\n";
  printf(msg, signum);  
  done = 1;
}

void report_and_exit(const char* msg) {
  perror(msg);
  exit(-1);
}

int main() {
  struct sigaction act;
  act.sa_handler = signal_handler;
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &act, 0);

  struct timespec ts;
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
  sem_t* semptr = sem_open(SEM_NAME, O_RDWR);
  if (semptr == (void*) -1) report_and_exit("sem_open()");
  while (!done) {
    printf("sem_wait()'ing\n");
    if (sem_wait(semptr) < 0) report_and_exit("sem_wait()");
    timespec_get(&ts, TIME_UTC);
    printf("sem_wait()'ed at %ld.%09ld\n", ts.tv_sec, ts.tv_nsec);
    for (int i = 0; i < 256; i++)
      write(STDOUT_FILENO, memptr + i, 1); /* one byte at a time */
    printf("\n......%d bytes in shared memory truncated......\n", SHM_SIZE - 512);
    for (int i = SHM_SIZE - 256; i < SHM_SIZE; i++)
      write(STDOUT_FILENO, memptr + i, 1); /* one byte at a time */
    printf("\n");
    sem_post(semptr);
    sleep(5);
  }
  /* cleanup */
  munmap(memptr, SHM_SIZE);
  close(fd);
  sem_close(semptr);
  unlink(SEM_NAME);
  return 0;
}