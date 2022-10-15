#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <tchar.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "col_parser.c"

int main()
{
   void* fd;
   void* memptr;
   int* int_col_ptr = malloc(MAX_LINE * sizeof(int));
   double* dbl_col_ptr = malloc(MAX_LINE * sizeof(double));
   char* chr_col_ptr = malloc(MAX_LINE * sizeof(char) * CHAR_COL_BUF_SIZE);
   size_t int_col_count, dbl_col_count, chr_col_count;
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
   
   while (1)
   {
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
      memset(chr_col_ptr, 0, (MAX_LINE * sizeof(char) * CHAR_COL_BUF_SIZE));
      int_col_count = read_ints(".\\data\\col1_int.txt", int_col_ptr, MAX_LINE);
      dbl_col_count = read_dbls(".\\data\\col2_dbl.txt", dbl_col_ptr, MAX_LINE);
      chr_col_count = read_chrs(".\\data\\col3_chr.txt", chr_col_ptr, MAX_LINE);

      if (int_col_count != dbl_col_count || dbl_col_count != chr_col_count) {
         fprintf(stderr, "Columns have different size, this is not supported!\n");
         UnmapViewOfFile(memptr);
         CloseHandle(fd);
         CloseHandle(sem_ptr);
         return 1;
      }
      
      memcpy(memptr, &int_col_count, sizeof(size_t));
      memcpy((char*)memptr + sizeof(size_t), int_col_ptr, sizeof(int) * int_col_count);
      memcpy((char*)memptr + sizeof(size_t) + sizeof(int) * int_col_count, dbl_col_ptr, sizeof(double) * dbl_col_count);
      memcpy((char*)memptr + sizeof(size_t) + sizeof(int) * int_col_count+ sizeof(double) * dbl_col_count, chr_col_ptr, sizeof(char) * chr_col_count * CHAR_COL_BUF_SIZE);
      
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
      printf("read from files...\n");
      Sleep(5000);
   }

   
   unsigned long long t0 = get_timestamp_100nano();
   printf("Unlocked at %lld, shm_reader should be able to read now.\n", t0);
   UnmapViewOfFile(memptr);

   CloseHandle(fd);
   CloseHandle(sem_ptr);
   free(int_col_ptr);
   free(dbl_col_ptr);
   free(chr_col_ptr);
   return 0;
}