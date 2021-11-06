#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 1000000

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

  struct timeval stop, start;
  FILE *fin, *fout;
  int arr[SIZE];
  int i;

  fin = fopen("quicksort.in", "r");
  for (i = 0; i < SIZE; i++)
    fscanf(fin, "%d", &arr[i]);
  fclose(fin);

  gettimeofday(&start, NULL);
  quicksort(arr, 0, SIZE - 1);
  gettimeofday(&stop, NULL);
  printf("%.3g sec\n", ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000.0 / 1000.0);

  fout = fopen("quicksort.out.c", "w");
  /* Creates an empty file for writing. If a file with the same
     name already exists, its content is erased and the file is
     considered as a new empty file. */
  for (i = 0; i < SIZE; i++)
    fprintf(fout, "%d, ", arr[i]);
  fclose(fout);

  return 0;
}