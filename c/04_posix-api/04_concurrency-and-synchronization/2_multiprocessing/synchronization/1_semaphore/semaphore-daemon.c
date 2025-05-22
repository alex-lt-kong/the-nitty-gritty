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

volatile sig_atomic_t ev_flag = 0;
sem_t* semptr;

void signal_handler(int signum) {
  char msg[] = "Signal %d received by signal_handler(), press Enter to send the signal to event loop\n";
  printf(msg, signum);  
  ev_flag = 1;
}

int main() {
  struct sigaction act;
  act.sa_handler = signal_handler;
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &act, 0);

  /* semaphore code to lock the shared mem */
  semptr = sem_open(SEM_NAME, O_CREAT, SEM_PERMS, INITIAL_VALUE);
  struct timespec ts;
  if (semptr == (void*) -1) {
    perror("sem_open()");
    return 1;
  }
  while (!ev_flag) {
    printf("Press Enter to sem_wait() for the critical section...\n");
    // if (getchar() == '\n') {}
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);

    struct timespec timeout = {5, 0}; // Wait up to 5 seconds (adjust as needed)

    int ret = pselect(STDIN_FILENO + 1, &fds, NULL, NULL, &timeout, NULL);

    if (ret > 0) {
      if (getchar() == '\n') {}; // Normal input handling
    } else if (ret == -1 && ev_flag) {
      break; // Exit when signal received
    } else
      continue;



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
    printf("^^^^^ sem_post()'ed, lock released at: %ld.%09ld\n", ts.tv_sec, ts.tv_nsec);
  }
  printf("event loop exited\n");
  if (semptr != NULL) {
    sem_close(semptr);
    semptr = NULL;
  }
  sem_unlink(SEM_NAME);
  return 0;
}