#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>

#include "common.h"
#include "col_parser.c"

volatile int done = 0;

static void signal_handler(int sig_num) {
    printf("Signal %d received, program will exit in a sec...\n", sig_num);
    done = 1;  
}


int main()
{
    void* fd;
    void* memptr;
    int* int_col_ptr = malloc(MAX_LINE_COUNT * sizeof(int));
    double* dbl_col_ptr = malloc(MAX_LINE_COUNT * sizeof(double));
    char* chr_col_ptr = malloc(MAX_LINE_COUNT * sizeof(char) * CHAR_COL_BUF_SIZE);
    size_t int_col_line_count, dbl_col_line_count, chr_col_line_count, line_count;
    // Create a semaphore
    void* sem_ptr = CreateSemaphore( 
        NULL,           // default security attributes
        MAX_SEM_COUNT,  // initial count
        MAX_SEM_COUNT,  // maximum count
        sem_name);      // the name of the semaphore
    if (sem_ptr == NULL) {
        fprintf(stderr, "CreateSemaphore error: %ld\n", GetLastError());
        return 1;
    }

    fd = CreateFileMapping(
                    INVALID_HANDLE_VALUE,    // use paging file
                    NULL,                    // default security
                    PAGE_READWRITE,          // read/write access
                    0,                       // maximum object size (high-order DWORD)
                    SHM_SIZE,                // maximum object size (low-order DWORD)
                    shm_name);                 // name of mapping object

    if (fd == NULL) {
        fprintf(stderr, "Could not create file mapping object (%ld).\n",  GetLastError());
        return 1;
    }
    memptr = (void*) MapViewOfFile(fd,   // handle to map object
                        FILE_MAP_ALL_ACCESS, // read/write permission
                        0,
                        0,
                        SHM_SIZE);

    if (memptr == NULL) {
        fprintf(stderr, "Could not map view of file (%ld).\n", GetLastError());
        CloseHandle(fd);
        return 1;
    }
    // handle OS signals
    signal(SIGABRT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);
    while (!done) {
        Sleep(1000);
        int retval = WaitForSingleObject(
            sem_ptr,
            3600 * 1000L  // wait for at most 1 hour
        );
        if (retval != WAIT_OBJECT_0) {
            UnmapViewOfFile(memptr);
            CloseHandle(fd);
            CloseHandle(sem_ptr);
            fprintf(stderr, "Unknow return value: %d\n", retval);
            return 1;
        }
        memset(chr_col_ptr, 0, (MAX_LINE_COUNT * sizeof(char) * CHAR_COL_BUF_SIZE));
        int_col_line_count = read_ints(".\\data\\col1_int.txt", int_col_ptr);
        dbl_col_line_count = read_dbls(".\\data\\col2_dbl.txt", dbl_col_ptr);
        chr_col_line_count = read_chrs(".\\data\\col3_chr.txt", chr_col_ptr);
        
        if (int_col_line_count != dbl_col_line_count || dbl_col_line_count != chr_col_line_count) {
            fprintf(
                stderr,
                "[%lf] Columns have different line_count, the result is not well-defined. New data will not be written into shared memory "
                "(old data, if any, are still in shared memory and intact.)\n",
                get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0
            );
        } else {
            line_count = int_col_line_count;
            size_t content_size = sizeof(line_count) + (sizeof(int) + sizeof(double) + sizeof(char) * CHAR_COL_BUF_SIZE) * line_count;
            if (SHM_SIZE < content_size) {
                fprintf(
                    stderr,
                    "[%lf] Shared memory too small for incoming data. New data will not be written into shared memory "
                    "(old data, if any, are still in shared memory and intact.)\n",
                    get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0
                );
            }else {
                memcpy(memptr, &line_count, sizeof(size_t));   
                memcpy((unsigned char*)memptr + sizeof(line_count), int_col_ptr, sizeof(int) * line_count);
                memcpy((unsigned char*)memptr + sizeof(line_count) + sizeof(int) * line_count, dbl_col_ptr, sizeof(double) * line_count);
                memcpy((unsigned char*)memptr + sizeof(line_count) + sizeof(int) * line_count+ sizeof(double) * line_count, chr_col_ptr, sizeof(char) * line_count * CHAR_COL_BUF_SIZE);
                printf(
                    "[%lf] %llu KB of data loaded from files, shared memory size is %u KB\n",
                    get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0, content_size / 1024, SHM_SIZE / 1024
                );
            }
        }
        if (!ReleaseSemaphore(
            sem_ptr,  // handle to semaphore
            1,        // increase count by one
            NULL)     // not interested in previous count
        ) {
            fprintf(stderr, "ReleaseSemaphore() error: %ld\n", GetLastError());
            UnmapViewOfFile(memptr);
            CloseHandle(fd);
            CloseHandle(sem_ptr);
            return 1;
        }
        
    }


    UnmapViewOfFile(memptr);

    CloseHandle(fd);
    CloseHandle(sem_ptr);
    free(int_col_ptr);
    free(dbl_col_ptr);
    free(chr_col_ptr);
    
    printf("Program exited gracefully\n");
    return 0;
}
