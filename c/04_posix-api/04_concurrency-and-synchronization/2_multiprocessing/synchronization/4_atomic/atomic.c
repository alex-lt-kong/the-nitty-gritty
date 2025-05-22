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

int cnt = 0;
sig_atomic_t cnt_sig_atomic = 0;
_Atomic int cnt_atomic_qual = 0;  // qual: type qualifier
_Atomic(int) cnt_atomic_spec = 0; // spec: type specifier

void *adding() {
  for (int i = 0; i < 10000; i++) {
    ++cnt_atomic_qual;
    ++cnt_atomic_spec;
    ++cnt;
    ++cnt_sig_atomic;
  }
  return NULL;
}

void test_concurrent_addition() {
  printf("test_concurrent_addition():\n");
  pthread_t tid[10];
  for (int i = 0; i < 10; i++)
    if (pthread_create(&tid[i], NULL, adding, NULL) != 0) {
      perror("pthread_create():");
      return;
    }
  for (int i = 0; i < 10; i++)
    pthread_join(tid[i], NULL);
  printf("the value of acnt_c is %d\n", cnt_atomic_qual);
  printf("the value of acnt_cpp is %d\n", cnt_atomic_spec);
  printf("the value of cnt is %d\n", cnt);
  printf("the value of scnt is %d\n", cnt_sig_atomic);
}

#define THREAD_NUM 5000
#define PASS_NUM 128
_Atomic uint8_t *table = NULL;
_Atomic size_t thread_counter = 0;
_Atomic short error = 0;

void *counting(void *pass) {
  uint8_t p = *((uint8_t *)pass);
  if (error) {
    return NULL;
  }
  int execution_id = (thread_counter++) % THREAD_NUM;
  // Alternative implementation, breaks atomicity:
  // thread_counter = (thread_counter + 1) % THREAD_NUM;
  // int execution_id = thread_counter;
  if (table[execution_id] == p) {
    table[execution_id] = p + 1;
  } else {
    fprintf(stderr, "ERROR: table[%d]==%u, expecting %u\n", execution_id,
            table[execution_id], p);
    ++error;
  }
  return NULL;
}

void test_concurrent_counting() {

  printf("test_concurrent_counting():\n");
  table = (_Atomic uint8_t *)calloc(THREAD_NUM, sizeof(short));
  if (table == NULL) {
    perror("calloc()");
    goto err_calloc_table;
  }
  pthread_t *ths = (pthread_t *)calloc(THREAD_NUM, sizeof(pthread_t));
  if (table == NULL) {
    perror("malloc()");
    goto err_malloc_pthread;
  }

  for (uint8_t i = 0; i < PASS_NUM && error == 0; ++i) {
    printf("Pass no.%u\n", i);
    int j = 0;
    for (; j < THREAD_NUM; ++j) {
      if (pthread_create(&ths[j], NULL, counting, &i) != 0) {
        --j;
        perror("pthread_create()");
        break;
      }
    }
    for (int k = 0; k <= j; ++k) {
      (void)pthread_join(ths[k], NULL);
    }
  }

  free(ths);
err_malloc_pthread:
  free(table);
err_calloc_table:
  return;
}

int main(void) {
  printf("is_sig_atomic_t_just_int(): %d\n\n", is_sig_atomic_t_just_int());
  test_concurrent_addition();
  printf("\n");
  test_concurrent_counting();
  printf("\n");
  return 0;
}