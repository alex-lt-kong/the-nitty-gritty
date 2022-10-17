#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>

#include "common.h"
#include "col_parser.c"

char sample_dt[][21] = {
    "2021-10-21T12:17:14Z",
    "2021-09-12T21:21:24Z",
    "2021-08-31T20:33:34Z",
    "2020-07-01T22:46:44Z",
    "2020-06-27T12:56:54Z",
    "2020-05-04T13:04:64Z",
    "2022-04-29T14:12:22Z",
    "2022-03-13T15:29:30Z",
    "2022-02-14T16:30:46Z",
    "2022-01-14T16:49:52Z"
};

char sample_strings[][256] = {
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed doeiusmod tempor incididunt ut labore et dolore magna aliqua. Ut",
    "enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure",
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non",
    "proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    "In publishing and graphic design, Lorem ipsum is a placeholder text commonly used to demonstrate the visual form of a document or a typeface without relying on meaningful content.",
    "ipsum may be used as a placeholder before final copy is available. It is also used to temporarily replace text in a process called greeking, which allows designers to consider the form of a webpage or publication.",
    "typically a corrupted version of De finibus bonorum et malorum, a 1st-century BC text by the Roman statesman and philosopher Cicero, with words altered, added, and removed to make it nonsensical and improper Latin.",
    "Versions of the Lorem ipsum text have been used in typesetting at least since the 1960s, when it was popularized by advertisements for Letraset transfer sheets.[1]",
    "Other popular word processors, including Pages and Microsoft Word, have since adopted Lorem ipsum,[2] as have many LaTeX packages,[3][4][5] web content managers such as Joomla! and WordPress, and CSS libraries such as Semantic UI.[6]",
    "Sed ut perspiciatis, unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam eaque ipsa"
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
        memoffset = rnd_int;
        memset((unsigned char*)memptr + sizeof(line_count) + line_count * sizeof(int) + memoffset * char_col_size, 0, char_col_size);
        memset((unsigned char*)memptr + sizeof(line_count) + line_count * (sizeof(int) + char_col_size + sizeof(double)) + memoffset * char_col_size, 0, char_col_size);

        memcpy((unsigned char*)memptr, &line_count, sizeof(line_count));   
        memcpy((unsigned char*)memptr + sizeof(line_count) + memoffset * sizeof(int), &rnd_int, sizeof(int));
        memcpy((unsigned char*)memptr + sizeof(line_count) + line_count * sizeof(int) + memoffset * char_col_size, sample_dt[rand() % 9], char_col_size);
        memcpy((unsigned char*)memptr + sizeof(line_count) + line_count * (sizeof(int) + char_col_size) + memoffset * sizeof(double), &rnd_dbl, sizeof(double));
        memcpy((unsigned char*)memptr + sizeof(line_count) + line_count * (sizeof(int) + char_col_size + sizeof(double)) + memoffset * char_col_size, sample_strings[rand() % 9], char_col_size);
        
        if (iter_count % (1000 * 1000) == 0) {
            QueryPerformanceCounter(&t1);
            printf("Takes %6.1lf ms to write 1M rows (%.1lf / %.1lf MB used)\n",
            (t1.QuadPart - t0.QuadPart) * 1e6 / freq.QuadPart / 1000.0,
            content_size / 1024.0 / 1024.0, SHM_SIZE / 1024.0 / 1024.0);
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
