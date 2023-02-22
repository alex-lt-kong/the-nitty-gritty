#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

pthread_mutex_t mutex;
struct timespec ts;
volatile int done = 0;

void* dummy_thread0() {
  while (1) {
    printf("Thread0 | press Enter to pthread_mutex_lock() or any other key to exit...\n");
    if (getchar() == '\n') {
      printf("Thread0 | pthread_mutex_lock()'ing...\n");
      pthread_mutex_lock(&mutex);
      printf("Thread0 | pthread_mutex_lock()'ed\n");
      printf("Thread0 | press any key to pthread_mutex_unlock()\n");
      getchar();
      pthread_mutex_unlock(&mutex);
      timespec_get(&ts, TIME_UTC);
      printf("Thread0 | pthread_mutex_unlock()'ed at %ld.%09ld\n", ts.tv_sec, ts.tv_nsec);
    } else {
      break;
    }
  }
  printf("Thread0 | exiting\n");
  done = 1;
  return NULL;
}

void* dummy_thread1() {
  while (!done) {
    printf("Thread1 | waiting for 5 sec to pthread_mutex_lock()\n");
    sleep(5);
    printf("Thread1 | pthread_mutex_lock()'ing...\n");
    pthread_mutex_lock(&mutex);
    timespec_get(&ts, TIME_UTC);
    printf("Thread1 | pthread_mutex_lock()'ed at %ld.%09ld\n", ts.tv_sec, ts.tv_nsec);
    printf("Thread1 | waiting for 20 sec and then pthread_mutex_unlock()'ed\n");
    sleep(20);
    pthread_mutex_unlock(&mutex);
    printf("Thread1 | pthread_mutex_unlock()'ed\n");
  }
  printf("Thread1 | exiting\n");
  return NULL;
}

int main() {
  pthread_mutex_init(&mutex, NULL);
  pthread_t tid0, tid1;
  pthread_create(&tid0, NULL, dummy_thread0, NULL);
  pthread_create(&tid1, NULL, dummy_thread1, NULL);
  pthread_join(tid0, NULL);
  printf("Main    | Thread0 exitied\n");
  pthread_join(tid1, NULL);
  printf("Main    | Thread1 exitied\n");
  pthread_mutex_destroy(&mutex);
  return 0;
}