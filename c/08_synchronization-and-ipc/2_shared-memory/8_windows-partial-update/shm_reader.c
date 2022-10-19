#include <stdio.h>
#include <windows.h>
#include <stdio.h>

#include "common.h"

int read_shm(int** int_arr_ptr, char** dt_arr_ptr, double** dbl_arr_ptr, char** chr_arr_ptr, size_t hi, size_t lo)
{

    if (lo > hi || hi >= MAX_LINE_COUNT) {
        fprintf(stderr, "invalid combination of hi, lo: %llu, %llu\n", hi, lo);
        return 0;
    }
    LARGE_INTEGER freq, t0, t1, t2, t3;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    size_t total_line_count, slice_line_count;
    void* fd;
    void* memptr;
    void* sem_ptr = CreateSemaphore(NULL, MAX_SEM_COUNT, MAX_SEM_COUNT, sem_name);      

    fd = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, shm_name);

    if (fd == NULL) {
        fprintf(
            stderr, 
            "[%.6lf] OpenFileMapping() failed, "
            "either shared memory is not ready or the program doesn't have access permission. (Error code: %ld).\n",
            get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0, GetLastError()
        );
        CloseHandle(sem_ptr);
        return 0;
    }

   memptr = (void*) MapViewOfFile(fd, FILE_MAP_READ, 0, 0, SHM_SIZE);

   if (memptr == NULL) {
      fprintf(
        stderr,
        "[%.6lf] MapViewOfFile()@shm_reader.so failed (Error code: %ld).\n",
        get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0, GetLastError());
      CloseHandle(fd);
      CloseHandle(sem_ptr);
      return 0;
   }
   
    int retval = WaitForSingleObject(sem_ptr, 3600 * 1000L);
    if (retval != WAIT_OBJECT_0) {
        UnmapViewOfFile(memptr);
        CloseHandle(fd);
        CloseHandle(sem_ptr);
        fprintf(stderr, "WaitForSingleObject()@shm_reader.so failed (Return value: %d)\n", retval);
        return 0;
    }
    QueryPerformanceCounter(&t1);
    memcpy(&total_line_count, memptr, sizeof(size_t));
    if (total_line_count > MAX_LINE_COUNT) {
        fprintf(stderr, "Shared memory content is larger then allocated memory, won't write\n");
        hi = -1;
        lo = 0;
    } else {
        slice_line_count = (hi - lo + 1);
        (*int_arr_ptr) = malloc(MAX_LINE_COUNT * sizeof(int));
        (*dt_arr_ptr) = calloc(MAX_LINE_COUNT, CHAR_COL_BUF_SIZE);
        (*dbl_arr_ptr) = malloc(MAX_LINE_COUNT * sizeof(double));
        (*chr_arr_ptr) = calloc(MAX_LINE_COUNT, CHAR_COL_BUF_SIZE);
        memcpy(*int_arr_ptr, (char*)memptr + sizeof(size_t) + lo * sizeof(int),  slice_line_count * sizeof(int));
        memcpy( *dt_arr_ptr, (char*)memptr + sizeof(size_t) + total_line_count * (sizeof(int)                                 ) + lo * char_col_size, slice_line_count * char_col_size);
        memcpy(*dbl_arr_ptr, (char*)memptr + sizeof(size_t) + total_line_count * (sizeof(int) + char_col_size                 ) + lo * sizeof(double), slice_line_count * sizeof(double));
        memcpy(*chr_arr_ptr, (char*)memptr + sizeof(size_t) + total_line_count * (sizeof(int) + char_col_size + sizeof(double)) + lo * char_col_size, slice_line_count * char_col_size);
        QueryPerformanceCounter(&t2);
    }
    if (!ReleaseSemaphore(sem_ptr, 1, NULL)) {
        fprintf(stderr, "ReleaseSemaphore() error: %ld\n", GetLastError());
        hi = -1;
        lo = 0;
    }
    UnmapViewOfFile(memptr);   
    CloseHandle(fd);
    CloseHandle(sem_ptr);
    QueryPerformanceCounter(&t3);
    printf(
        "[%.6lf] read_shm()@shm_reader.so returned. (%.1lf + %.1lf + %.1lf = %.1lf us)\n",
        get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0,
        (t1.QuadPart - t0.QuadPart) * 1e6 / freq.QuadPart,
        (t2.QuadPart - t1.QuadPart) * 1e6 / freq.QuadPart,
        (t3.QuadPart - t2.QuadPart) * 1e6 / freq.QuadPart,
        (t3.QuadPart - t0.QuadPart) * 1e6 / freq.QuadPart
    );
    return hi - lo + 1;
}