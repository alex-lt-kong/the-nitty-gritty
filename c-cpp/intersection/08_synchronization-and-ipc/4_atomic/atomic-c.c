#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int is_sig_atomic_t_just_int() {
  // __builtin_types_compatible_p() is defined here:
  // https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005ftypes_005fcompatible_005fp
  int a;
  sig_atomic_t b;
  return (__builtin_types_compatible_p(typeof(a), typeof(b)) &&
          __builtin_types_compatible_p(typeof(b), int));
}

_Atomic int acnt = 0;
int cnt = 0;
sig_atomic_t scnt = 0;

void *adding(void *input) {
  for (int i = 0; i < 10000; i++) {
    ++acnt;
    ++cnt;
    ++scnt;
  }
  pthread_exit(NULL);
}

void test() {
  pthread_t tid[10];
  for (int i = 0; i < 10; i++)
    pthread_create(&tid[i], NULL, adding, NULL);
  for (int i = 0; i < 10; i++)
    pthread_join(tid[i], NULL);

  printf("the value of acnt is %d\n", acnt);
  printf("the value of cnt is %d\n", cnt);
  printf("the value of scnt is %d\n", scnt);
}

int main(void) {
  printf("is_sig_atomic_t_just_int(): %d\n", is_sig_atomic_t_just_int());
  test();
  return 0;
}