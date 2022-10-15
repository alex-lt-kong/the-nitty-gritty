#include <stdio.h>
#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <tchar.h>
#include "common.h"
#pragma comment(lib, "user32.lib")

int read_shm(int arr[], int length)
{
    for (int i = 0; i < length; ++i) {
        printf("%d\n", arr[i]);
    }
    void* fd;
    void* memptr;
    void* sem_ptr = CreateSemaphore( 
        NULL,           // default security attributes
        MAX_SEM_COUNT,  // initial count
        MAX_SEM_COUNT,  // maximum count
        sem_name);      // the name of the semaphore

   fd = OpenFileMapping(
                   FILE_MAP_ALL_ACCESS,   // read/write access
                   FALSE,                 // do not inherit the name
                   shm_name);               // name of mapping object

   if (fd == NULL) {
      fprintf(stderr, "Could not open file mapping object (%ld).\n", GetLastError());
      return 1;
   }

   memptr = (void*) MapViewOfFile(fd, // handle to map object
               FILE_MAP_ALL_ACCESS,  // read/write permission
               0,
               0,
               SHM_SIZE);

   if (memptr == NULL) {
      fprintf(stderr, "Could not map view of file (%ld).\n", GetLastError());
      CloseHandle(fd);
      return 1;
   }
   int retval = WaitForSingleObject(
      sem_ptr,
      3600 * 1000L  // wait for at most 1 hour
   );
   if (retval != WAIT_OBJECT_0) {
      UnmapViewOfFile(memptr);
      CloseHandle(fd);
      CloseHandle(sem_ptr);
      fprintf(stderr, "Unknow error!\n");
      return 1;
   }
   unsigned long long t = get_timestamp_100nano();
   printf("Lock entered at %lld\n", t);

   printf("%s\n", (char*) memptr);
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
   for (int i = 0; i < length; ++i) {
      arr[i] += 2;
   }
   return 666;
}