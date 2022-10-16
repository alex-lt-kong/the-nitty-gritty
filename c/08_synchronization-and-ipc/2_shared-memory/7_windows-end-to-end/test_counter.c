#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    LARGE_INTEGER freq;
    LARGE_INTEGER begin_time;
    LARGE_INTEGER end_time;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&begin_time);
    Sleep(1);
    QueryPerformanceCounter(&end_time);
    int time_micro = (double)(end_time.QuadPart - begin_time.QuadPart) * 1e6 / (double)freq.QuadPart;
    printf("%d\b", time_micro);
    return 0;
}
