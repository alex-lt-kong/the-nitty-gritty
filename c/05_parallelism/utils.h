#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

inline uint64_t get_timestamp_in_microsec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1000000 * tv.tv_sec + tv.tv_usec;
}
