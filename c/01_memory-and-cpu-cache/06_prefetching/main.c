#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define ARR_SIZE (512 * 1024 * 1024)
#define NUM_LOOKUPS (8 * 1024 * 1024)
#define DO_PREFETCH

int binary_search(int *array, int key, int enable_prefetch) {
  int low = 0, high = ARR_SIZE - 1, mid;
  while(low <= high) {
    mid = (low + high) / 2;

    if (enable_prefetch) {
      // prefetch the possible mid of the next iteration
      __builtin_prefetch (&array[(mid + 1 + high) / 2], 0, 1);
      __builtin_prefetch (&array[(low + mid - 1) / 2], 0, 1);
    }

    if(array[mid] < key)
      low = mid + 1; 
    else if(array[mid] == key)
      return mid;
    else if(array[mid] > key)
      high = mid - 1;
  }
  return mid;
}

int main() {

  uint32_t sum = 0;
  struct timespec ts;
  double t0, delta;
  int *array =  malloc(ARR_SIZE*sizeof(int));
  for (int i = 0; i < ARR_SIZE; ++i){
    array[i] = i;
  }

  srand(time(NULL));
  int *lookups = malloc(NUM_LOOKUPS * sizeof(int));
  for (int i = 0; i < NUM_LOOKUPS; ++i) {
    lookups[i] = rand() % ARR_SIZE;
  }

  timespec_get(&ts, TIME_UTC);
  t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
  for (int i = 0; i < NUM_LOOKUPS; i++) {
    sum += binary_search(array, lookups[i], 1);
  }
  timespec_get(&ts, TIME_UTC);
  delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
  printf("Prefetching enabled: %0.3lfsec\n", delta);

  timespec_get(&ts, TIME_UTC);
  t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
  for (int i = 0; i < NUM_LOOKUPS; i++) {
    sum += binary_search(array, lookups[i], 0);
  }
  timespec_get(&ts, TIME_UTC);
  delta = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0 - t0;
  printf("Prefetching disabled: %0.3lfsec\n", delta);

  free(array);
  free(lookups);
  return sum;
}