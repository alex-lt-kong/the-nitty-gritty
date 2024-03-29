#include <stdio.h>
#include <windows.h>
#include <stdio.h>

#include "common.h"

int read_shm(int* int_arr_ptr, int* dbl_arr_ptr,  char* chr_arr_ptr, int length)
{
    LARGE_INTEGER freq, t0, t1, t2;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    //printf("[%.6lf] read_shm()@shm_reader.so called\n", get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0);
    size_t line_count;
    void* fd;
    void* memptr;
    void* sem_ptr = CreateSemaphore( 
        NULL,           // default security attributes
        MAX_SEM_COUNT,  // initial count
        MAX_SEM_COUNT,  // maximum count
        sem_name        // the name of the semaphore
    );      

    fd = OpenFileMapping(
        FILE_MAP_ALL_ACCESS,   // read/write access
        FALSE,                 // do not inherit the name
        shm_name               // name of mapping object
    );

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

   memptr = (void*) MapViewOfFile(fd, // handle to map object
               FILE_MAP_ALL_ACCESS,  // read/write permission
               0,
               0,
               SHM_SIZE);

   if (memptr == NULL) {
      fprintf(
        stderr,
        "[%.6lf] MapViewOfFile()@shm_reader.so failed (Error code: %ld).\n",
        get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0, GetLastError());
      CloseHandle(fd);
      CloseHandle(sem_ptr);
      return 0;
   }
   
   int retval = WaitForSingleObject( // the count of a semaphore object is decreased by one
      sem_ptr,
      3600 * 1000L  // wait for at most 1 hour
   );
   if (retval != WAIT_OBJECT_0) {
      UnmapViewOfFile(memptr);
      CloseHandle(fd);
      CloseHandle(sem_ptr);
      fprintf(stderr, "WaitForSingleObject()@shm_reader.so failed (Return value: %d)\n", retval);
      return 0;
   }
   QueryPerformanceCounter(&t1);
   memcpy(&line_count, memptr, sizeof(size_t));
   memcpy(int_arr_ptr, (char*)memptr + sizeof(size_t), line_count * sizeof(int));
   memcpy(dbl_arr_ptr, (char*)memptr + sizeof(size_t) + line_count * sizeof(int), line_count * sizeof(double));
   memcpy(chr_arr_ptr, (char*)memptr + sizeof(size_t) + line_count * sizeof(int) + line_count * sizeof(double), line_count * sizeof(char) * CHAR_COL_BUF_SIZE);
   /* Memory scheme
    * |8 bytes| all ints |         all doubles           |                                              all strings                                        |                    
    */
   if (!ReleaseSemaphore(
      sem_ptr,  // handle to semaphore
      1,        // increase count by one
      NULL)     // not interested in previous count
   ) {
      fprintf(stderr, "ReleaseSemaphore error: %ld\n", GetLastError());
   }
   UnmapViewOfFile(memptr);   
   CloseHandle(fd);
   CloseHandle(sem_ptr);
   QueryPerformanceCounter(&t2);
   printf(
      "[%.6lf] read_shm()@shm_reader.so returned, %.1lf micro before lock, %.1lf micro after lock\n",
      get_timestamp_100nano() / 10.0 / 1000.0 / 1000.0,
      (t1.QuadPart - t0.QuadPart) * 1e6 / freq.QuadPart,
      (t2.QuadPart - t1.QuadPart) * 1e6 / freq.QuadPart
   );
   return line_count;
}