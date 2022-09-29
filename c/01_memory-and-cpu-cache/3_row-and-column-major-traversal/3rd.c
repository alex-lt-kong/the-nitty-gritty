// gcc -o main.out main.c -O3
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    size_t dim[] = {
           10,
           20,
           50,
          100,
          200,
          500,
         1000,
         2000,
         5000,
        10000,
        20000,
    };
    const int nDim = sizeof dim/sizeof dim[0];
    size_t d, i, j, k;
    struct timespec ts, te; // start & end times

    // allocate, use and retain the largest array (vector) size.
    d = dim[ nDim - 1 ];
    uint32_t *arr_ptr = (uint32_t*)malloc( d * d * sizeof *arr_ptr );

    timespec_get( &ts, TIME_UTC ); // just for curiosity
    for( j = 0; j < d; ++j )
        for( k = 0; k < d; ++k )
            *(arr_ptr + (j * d) + k) = (j + k); // simple assignment
    timespec_get( &te, TIME_UTC );
    { 
        double tss = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
        double tse = te.tv_sec + te.tv_nsec / 1000.0 / 1000.0 / 1000.0;
        printf( "Initial time taken = %0.9lf\n", tse - tss );
    }

    printf(
        "Dim\tArraySize(KB)\t"
        "Row-Major Time\tRM Sample\t"
        "Col-Major Time\tCM Sample\t Difference\n" );

    for( i = 0; i < nDim; ++i ) {
        double tss, tse;

        d = dim[i];
        printf( "%5lu,\t%11lu,\t", d, d * d * sizeof *arr_ptr / 1024 );

        timespec_get( &ts, TIME_UTC );
        for( j = 0; j < d; ++j )
            for( k = 0; k < d; ++k )
                *(arr_ptr + (j * d) + k) += (j + k); // Row-Major thrashing
        timespec_get( &te, TIME_UTC );

        tss = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
        tse = te.tv_sec + te.tv_nsec / 1000.0 / 1000.0 / 1000.0;
        double deltaR = tse - tss;
        // the 'randomly selected' data presented as hex, not decimal
        printf( "%0.9lf,\t%08X,\t", deltaR, *(arr_ptr + ts.tv_sec % d * d + ts.tv_nsec % d) );

        timespec_get( &ts, TIME_UTC );
        for( j = 0; j < d; ++j )
            for( k = 0; k < d; ++k )
                *(arr_ptr + (k * d )+ j) += (j + k); // Col-Major thrashing
        timespec_get( &te, TIME_UTC );

        tss = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
        tse = te.tv_sec + te.tv_nsec / 1000.0 / 1000.0 / 1000.0;
        double deltaC = tse - tss;
        printf( "%0.9lf,\t%08X\t", deltaC, *(arr_ptr + ts.tv_sec % d * d + ts.tv_nsec % d) );

        printf( "%0.9lf\n", deltaR - deltaC ); // difference
    }

    free( arr_ptr ); // thank you for your service. you're free now/

    return 0;
}