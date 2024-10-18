#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct JobPayload {
  double *vec_a;
  double *vec_b;
  double *vec_c;
  int64_t arr_size;
  int64_t offset;
};

inline
#ifdef _WIN32
    DWORD
#else
    long
#endif
    get_cpu_cores() {
#ifdef _WIN32
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  return sysInfo.dwNumberOfProcessors;
#else
  return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}
