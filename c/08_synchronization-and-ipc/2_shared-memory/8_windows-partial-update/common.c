#include "common.h"

const char sem_name[] = "MySemaphore01";
const char shm_name[] = "Global\\AkFileMappingObject";
const size_t char_col_size = sizeof(char) * CHAR_COL_BUF_SIZE;

unsigned long long get_timestamp_100nano() {
   FILETIME ft;
   GetSystemTimeAsFileTime(&ft);
   unsigned long long tt = ft.dwHighDateTime;
   tt <<=32;
   tt |= ft.dwLowDateTime;
   tt -= 116444736000000000ULL;  //offset to January 1, 1970 (start of Unix epoch) in "ticks"
   return tt;
}

