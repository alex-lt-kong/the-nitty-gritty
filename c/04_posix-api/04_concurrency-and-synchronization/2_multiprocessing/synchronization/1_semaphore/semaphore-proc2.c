#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>
#include <time.h>
#include <signal.h>

#include "common.h"

volatile sig_atomic_t ev_flag = 0;
sem_t* semptr;

void signal_handler(int signum) {
  char msg[] = "Signal %d received by signal_handler(), the event loop will respond soon\n";
  printf(msg, signum);  
  ev_flag = 1;
}

int main() {

  int AccessPerms = 0644;
  struct timespec ts;

  struct sigaction act;
  act.sa_handler = signal_handler;
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &act, 0);
  sigaction(SIGTERM, &act, 0);

  /* create a semaphore for mutual exclusion */
  semptr = sem_open(SEM_NAME, O_RDWR);
  if (semptr == (void*) -1) {
    perror("sem_open()");
    return 1;
  }
  while (!ev_flag) {
    printf("Waiting for mutex..\n");
    /* use semaphore as a mutex (lock) by waiting for writer to increment it */
    if (sem_wait(semptr) < 0) {
      perror("sem_post()");
      break;
    } /* wait until semaphore != 0 */
    timespec_get(&ts, TIME_UTC);
    printf("Lock entered at: %ld.%09ld \n", ts.tv_sec, ts.tv_nsec);
    sem_post(semptr);
    printf("Lock quited, waiting for 5 sec before next attempt...\n");
    sleep(5);
  }
  sem_close(semptr);
  sem_unlink(SEM_NAME);
  // sem_close() releases the semaphore from the program, sem_unlink() releases the semaphore for the entire OS
  printf("event loop exited\n");
  return 0;
}