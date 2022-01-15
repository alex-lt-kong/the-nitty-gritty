#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000000

using namespace std;

int main() {

    FILE *fp;
    int upper = 2147483647;
    int iterCount = 10;
    srand(time(NULL));
    for (int i = 0; i < iterCount; i++) {
        fp = fopen(("./../quicksort.in/quicksort.in" + to_string(i)).c_str(), "w");
        for (int j = 0; j < SIZE; ++j) {
        fprintf(fp, "%d\n", rand() % upper);
        }
        fclose(fp);
    }
    return 0;
}
