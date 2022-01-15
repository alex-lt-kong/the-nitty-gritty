#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <chrono>
#include <iostream>

#define SIZE 1000000

using namespace std;

void quicksort(int arr[SIZE], int first, int last){

  int i, j, pivot, temp;

  if (first >= last) return;

  int idx = rand() % (last - first) + first;
  temp = arr[idx];
  arr[idx] = arr[first];
  arr[first] = temp;

  pivot = first;
  i = first;
  j = last;

  while(i < j) {
    while(arr[i] <= arr[pivot] && i < last) i++;
    while(arr[j] > arr[pivot]) j--;
      if(i < j) {
        temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
  }

  temp = arr[pivot];
  arr[pivot] = arr[j];
  arr[j] = temp;
  quicksort(arr, first, j-1);
  quicksort(arr, j+1, last);
}

int main(){

    FILE *fin, *fout;
    int iterCount = 10, arr[SIZE], averageElapsedMs = 0;

    for (int i = 0; i < iterCount; i++) {
        fin = fopen(("./../quicksort.in" + std::to_string(i)).c_str(), "r");
        // ./../: To accommodate Qt's shadow build function
        // a bit mix of C and C++...
        for (int j = 0; j < SIZE; j++)
            fscanf(fin, "%d", &arr[j]);
        fclose(fin);

        chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
        quicksort(arr, 0, SIZE - 1);
        chrono::high_resolution_clock::time_point finish = chrono::high_resolution_clock::now();
        chrono::microseconds microseconds = chrono::duration_cast<chrono::microseconds>(finish-start);
        averageElapsedMs += microseconds.count() / 1000;
        cout << i << "-th iteration: " << microseconds.count() / 1000 << "ms" << endl;

        // Just in case we would like to double check the results
        fout = fopen(("./../quicksort.out" + std::to_string(i)).c_str(), "w");
        /* Creates an empty file for writing. If a file with the same
         name already exists, its content is erased and the file is
         considered as a new empty file. */
        for (int j = 0; j < SIZE; j++)
            fprintf(fout, "%d, ", arr[j]);
        fclose(fout);
    }
    cout << "Average: " << averageElapsedMs / iterCount << "ms" << endl;
    return 0;
}
