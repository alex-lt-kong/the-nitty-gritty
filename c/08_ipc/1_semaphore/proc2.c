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

#define ByteSize 1024

int main() {

  int AccessPerms = 0644;
  

  /* create a semaphore for mutual exclusion */
  sem_t* semptr = sem_open(SEM_NAME, O_RDWR);
  if (semptr == (void*) -1) {
    perror("sem_open()");
    return 1;
  }
  while (1) {
    printf("Waiting for mutex..\n");
    /* use semaphore as a mutex (lock) by waiting for writer to increment it */
    if (sem_wait(semptr) < 0) {
      perror("sem_post()");
      break;
    } /* wait until semaphore != 0 */
    printf("Lock entered\n");
    sem_post(semptr);
    printf("Lock quited\n");
    
  }
  sem_close(semptr);
  return 0;
}