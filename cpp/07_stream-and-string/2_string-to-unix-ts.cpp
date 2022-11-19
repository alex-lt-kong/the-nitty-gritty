#include <stdio.h>
#include <time.h>
#include <iostream>

using namespace std;

/* 2010-11-04T23:23:01Z */
time_t zulu_time(const char *time_str)
{
    struct tm tm = { 0 };

    if (!strptime(time_str, "%Y-%m-%d %H:%M:%S", &tm))
        return (time_t)-1;

    return mktime(&tm) - timezone;
}

int main() {
  cout << zulu_time("2022-01-23 10:29:25.870768") << endl;
  return 0;
}