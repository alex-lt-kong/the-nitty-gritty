#ifndef COMMON_H
#define COMMON_H

#define MAX_SEM_COUNT 1

#define SHM_SIZE 65536
#define CHAR_COL_BUF_SIZE 128

const char sem_name[] = "MySemaphore01";
const char shm_name[] = "Global\\AkFileMappingObject";

unsigned long long get_timestamp_100nano() {
   FILETIME ft;
   GetSystemTimeAsFileTime(&ft);
   unsigned long long tt = ft.dwHighDateTime;
   tt <<=32;
   tt |= ft.dwLowDateTime;
   tt -= 116444736000000000ULL;  //offset to January 1, 1970 (start of Unix epoch) in "ticks"
   return tt;
}

#endif /* COMMON_H */