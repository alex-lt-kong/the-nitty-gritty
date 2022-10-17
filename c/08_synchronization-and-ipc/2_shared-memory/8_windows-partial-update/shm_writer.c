#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>

#include "common.h"
#include "col_parser.c"

char sample_dt[][20] = {
    "2022-10-11T22:33:14Z",
    "2022-09-11T22:33:24Z",
    "2022-08-11T22:33:34Z",
    "2022-07-11T22:33:44Z",
    "2021-10-11T12:33:54Z",
    "2020-10-11T12:34:64Z",
    "2022-10-12T12:35:74Z",
    "2022-10-13T12:36:84Z",
    "2022-10-14T12:37:94Z",
};

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
    const size_t line_count = MAX_LINE_COUNT;
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    srand(time(NULL));
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
    int retval = 0;
    const size_t line_size = (sizeof(int) + char_col_size + sizeof(double) + char_col_size);
    const size_t content_size = sizeof(line_count) + line_size * line_count;
    memset(memptr, 0, content_size);
    if (SHM_SIZE < content_size) {
        fprintf(stderr, "[%.6lf] Shared memory too small for incoming data. New data will not be written into shared memory ",
        get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0
        );
        done = 1;
    }
    size_t memoffset = 0;
    unsigned int iter_count = 0;
    QueryPerformanceCounter(&t0);
    while (!done) {
        ++ iter_count; 
        if ((retval = WaitForSingleObject(sem_ptr, 3600 * 1000L)) != WAIT_OBJECT_0) {
            done = 1;
            fprintf(stderr, "WaitForSingleObject() returns unknown value: %d\n", retval);
            break;
        }
        int rnd_int = rand() % MAX_LINE_COUNT;
        double rnd_dbl = ((double)rand() * (10 - (-10)) ) / (double)RAND_MAX + (-10);
        char rnd_str[] = "read_dbls(\".\\data\\col2_dbl.txt\", dbl_col_ptr)";
        memoffset = rnd_int;
        memset((unsigned char*)memptr + sizeof(line_count) + line_count * sizeof(int) + memoffset * char_col_size, 0, char_col_size);
        memset((unsigned char*)memptr + sizeof(line_count) + line_count * (sizeof(int) + char_col_size + sizeof(double)) + memoffset * char_col_size, 0, char_col_size);

        memcpy((unsigned char*)memptr, &line_count, sizeof(line_count));   
        memcpy((unsigned char*)memptr + sizeof(line_count) + memoffset * sizeof(int), &rnd_int, sizeof(int));
        memcpy((unsigned char*)memptr + sizeof(line_count) + line_count * sizeof(int) + memoffset * char_col_size, rnd_str, char_col_size);
        memcpy((unsigned char*)memptr + sizeof(line_count) + line_count * (sizeof(int) + char_col_size) + memoffset * sizeof(double), &rnd_dbl, sizeof(double));
        memcpy((unsigned char*)memptr + sizeof(line_count) + line_count * (sizeof(int) + char_col_size + sizeof(double)) + memoffset * char_col_size, rnd_str, char_col_size);
        
        if (iter_count % (1000 * 1000) == 0) {
            QueryPerformanceCounter(&t1);
            printf("Takes %.1lf ms to write 1,000,000 rows\n", (t1.QuadPart - t0.QuadPart) * 1e6 / freq.QuadPart / 1000.0);
            QueryPerformanceCounter(&t0);
        }

        if (!ReleaseSemaphore(sem_ptr, 1, NULL)) {
            fprintf(stderr, "ReleaseSemaphore() error: %ld\n", GetLastError());
            done = 1;
            break;
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
