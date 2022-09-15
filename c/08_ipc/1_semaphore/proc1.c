/** Compilation: gcc -o memwriter memwriter.c -lrt -lpthread **/
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
#include <stdint.h>

#include "common.h"

volatile sig_atomic_t done = 0;

int main() {
  /* semaphore code to lock the shared mem */
  sem_t* semptr = sem_open(SEM_NAME, O_CREAT, SEM_PERMS, INITIAL_VALUE);
  struct timespec ts;
  if (semptr == (void*) -1) {
    perror("sem_open()");
    return 1;
  }
  while (!done) {
    printf("Press any key to sem_wait() for the critical section or Q to exit the program...\n");
    if (getchar() == 'q') {
      break;
    }
    printf("sem_wait()'ing\n");
    if (sem_wait(semptr) < 0) {
      perror("sem_wait()");
      break;
    }
    printf("vvvvv CRITICAL SECTION entered! vvvvv\n");
    printf("Press Enter to quit the critical section\n");
    while (getchar() != '\n');
        
    timespec_get(&ts, TIME_UTC);    
    if (sem_post(semptr) < 0) {
      perror("sem_post()");
      break;
    }
    printf("^^^^^ sem_post()'ed, lock released at: %ld.%09ld UTC\n", ts.tv_sec, ts.tv_nsec);
  }
  printf("event loop exited\n");
  sem_close(semptr);
  return 0;
}