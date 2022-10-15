#define MAX_SEM_COUNT 1

#define SHM_SIZE 4096

const char sem_name[] = "MySemaphore01";
const char shm_name[] = "Global\\AkFileMappingObject";

unsigned long long get_timestamp_100nano() {
   FILETIME ft;
   GetSystemTimeAsFileTime(&ft);
   unsigned long long tt = ft.dwHighDateTime;
   tt <<=32;
   tt |= ft.dwLowDateTime;
   //tt /=10;
   tt -= 116444736000000000ULL;
   return tt;
}

unsigned long long get_timestamp_100nano_new()
{
   //Get the number of seconds since January 1, 1970 12:00am UTC
   //Code released into public domain; no attribution required.

   const unsigned long long UNIX_TIME_START = 0x019DB1DED53E8000; //January 1, 1970 (start of Unix epoch) in "ticks"
   const unsigned long long TICKS_PER_SECOND = 10000000; //a tick is 100ns

   FILETIME ft;
   GetSystemTimeAsFileTime(&ft); //returns ticks in UTC

   //Copy the low and high parts of FILETIME into a LARGE_INTEGER
   //This is so we can access the full 64-bits as an Int64 without causing an alignment fault
   LARGE_INTEGER li;
   li.LowPart  = ft.dwLowDateTime;
   li.HighPart = ft.dwHighDateTime;
 
   //Convert ticks since 1/1/1970 into seconds
   return (li.QuadPart - UNIX_TIME_START);
}