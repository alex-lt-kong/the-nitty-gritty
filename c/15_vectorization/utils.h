#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

unsigned long long get_timestamp_now();

double standard_deviation(double* arr, size_t arr_len, bool is_sample);
