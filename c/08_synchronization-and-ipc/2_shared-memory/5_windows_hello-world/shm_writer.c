#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <tchar.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"

int main()
{
   void* fd;
   void* memptr;
   char shm_boundary[128];
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
   // |                4096bytes              |
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
      fprintf(stderr, "Unknow return value: %d\n", retval);
      return 1;
   }
   unsigned long long t0, t1;
   t0 = get_timestamp_100nano();
   sprintf(shm_boundary, "\n========== Shared memory buffer BEGIN at %lld ==========\n",t0);
   // shm_boundary = "\n========== Shared memory buffer BEGIN at " + t0 + " ==========\n");
   strcpy((unsigned char*)memptr, shm_boundary);
   memset((unsigned char*)memptr + strlen(shm_boundary), 'Y', SHM_SIZE - (strlen(shm_boundary) + 1));

   t1 = get_timestamp_100nano();   
   sprintf(shm_boundary, "\n========== Shared memory buffer END at %lld ==========\n", t1);
   strcpy((unsigned char*)memptr + SHM_SIZE - (strlen(shm_boundary) + 1), shm_boundary);
   printf("shared memory memset()'ed, press any key to release the lock\n");

   getchar();

   if (!ReleaseSemaphore(
      sem_ptr,  // handle to semaphore
      1,        // increase count by one
      NULL)     // not interested in previous count
   ) {
      fprintf(stderr, "ReleaseSemaphore error: %ld\n", GetLastError());
      UnmapViewOfFile(memptr);
      CloseHandle(fd);
      CloseHandle(sem_ptr);
      return 1;
   }
   t1 = get_timestamp_100nano();
   printf("Unlocked at %lld, shm_reader should be able to read now.\n", t1);
   UnmapViewOfFile(memptr);

   CloseHandle(fd);
   CloseHandle(sem_ptr);
   return 0;
}