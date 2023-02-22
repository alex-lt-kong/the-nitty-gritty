#ifndef COMMON_H
#define COMMON_H

#include <windows.h>

#define MAX_SEM_COUNT 1

#define SHM_SIZE 4194304 // 4MB
#define CHAR_COL_BUF_SIZE 64
#define MAX_LINE_COUNT 32768

extern const char sem_name[];
extern const char shm_name[];

unsigned long long get_timestamp_100nano();
#endif /* COMMON_H */