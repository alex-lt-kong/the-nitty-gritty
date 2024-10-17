#include <stdint.h>
#include <stdlib.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

inline uint64_t get_timestamp_in_microsec() {
#ifdef _WIN32
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);

  ULARGE_INTEGER li;
  li.LowPart = ft.dwLowDateTime;
  li.HighPart = ft.dwHighDateTime;

  // Convert to microseconds
  // Windows file time is in 100-nanosecond intervals since January 1, 1601
  // (UTC) First, convert to microseconds, then adjust the epoch
  return (li.QuadPart / 10) - 11644473600000000ULL;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1000000 * tv.tv_sec + tv.tv_usec;
#endif
}