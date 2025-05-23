#include "semaphore-common.h"

#include <fcntl.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

volatile sig_atomic_t ev_flag = 0;

void signal_handler(const int signum) {
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + (char)(signum / 10);
  msg[9] = '0' + (char)(signum % 10);
  write(STDOUT_FILENO, msg, strlen(msg));
  ev_flag = 1;
}

int main(const int, char *argv[]) {
  struct timespec ts;
  struct sigaction act;
  act.sa_handler = signal_handler;
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_RESETHAND;
  if (sigaction(SIGINT, &act, 0) != 0 || sigaction(SIGTERM, &act, 0) != 0) {
    perror("sigaction()");
    goto err_sigaction;
  }

  sem_t *semptr = sem_open(SEM_NAME, O_CREAT, SEM_PERMS, INITIAL_VALUE);
  if (semptr == SEM_FAILED) {
    perror("sem_open()");
    goto err_sem_open;
  }
  while (!ev_flag) {
    printf("Waiting for semaphore..\n");
    if (sem_wait(semptr) < 0) {
      perror("sem_post()");
      break;
    } /* wait until semaphore != 0 */
    timespec_get(&ts, TIME_UTC);
    const int sleep_sec = 5;
    printf("sem_wait()'ed at: %ld.%09ld. Will wait for %d sec then leave "
           "the CRITICAL SECTION\n",
           ts.tv_sec, ts.tv_nsec, sleep_sec);
    sleep(sleep_sec);
    sem_post(semptr);
    printf("sem_post()'ed (i.e., left the CRITICAL SECTION)\n");
    // We must give the OS time to respond, if we skip nanosleep(),
    // semaphore-on-daemon will never be able to sem_wait() successfully
    ts.tv_sec = 0;
    ts.tv_nsec = 1;
    nanosleep(&ts, NULL);
  }
  sem_close(semptr);
  sem_unlink(SEM_NAME);
  // sem_close() releases the semaphore from the program, sem_unlink()
  // releases the semaphore for the entire OS
err_sem_open:
err_sigaction:
  printf("%s exited gracefully\n", argv[0]);
  return 0;
}
