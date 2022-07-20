#include "stdlib.h"
#include "stdio.h"
#include <iostream>

#define SIZE 1024 * 1024
// int: 4 Bytes
// >>> 1024 * 1024 * 4 / 1024 / 1024
// 4.0 MBytes

using namespace std;

void allocateHeapMemory(bool releaseAfterUse) {
  int *ptr = new int[SIZE];
  for (int i = 0; i < SIZE; i ++) {
    ptr[i] = i;
  }
  if (releaseAfterUse) {
    delete[] ptr;
    // The delete operator deallocates memory and calls the destructor for a single object created with new.
    // The delete [] operator deallocates memory and calls destructors for an array of objects created with new [].
  }
}

void allocateStackMemory() {
  int arr[SIZE];
  for (int i = 0; i < SIZE; i ++) {
    arr[i] = i;
  }
}

int main() {
  for (int i = 0; i < 100; i++) {
    allocateStackMemory();
  }
  cout << "stack memory allocated, memory usage reported by pmap:" << endl;
  system("pmap $(pgrep 'memory-leakage') | grep total");
  
  for (int i = 0; i < 100; i++) {
    allocateHeapMemory(false);
  }
  cout << "heap memory allocated and NOT delete[]'ed, memory usage reported by pmap:" << endl;
  system("pmap $(pgrep 'memory-leakage') | grep total");
  
  for (int i = 0; i < 100; i++) {
    allocateStackMemory();
  }
  cout << "stack memory allocated, memory usage reported by pmap:" << endl;
  system("pmap $(pgrep 'memory-leakage') | grep total");

  for (int i = 0; i < 100; i++) {
    allocateHeapMemory(true);
  }
  cout << "heap memory allocated and delete[]'ed, memory usage reported by pmap:" << endl;
  system("pmap $(pgrep 'memory-leakage') | grep total");
  // Usually you notice that delete[] does not release all the memory.
  // According to this link:
  // https://stackoverflow.com/questions/17008180/c-delete-does-not-free-all-memory-windows
  // it is because the OS may not reclaim all the virtual memory just in case 
  // the program needs them again soon.
  return 0;
}

/*
> ./memory-leakage 
stack memory allocated, memory usage reported by pmap:
 total             9540K
heap memory allocated and NOT delete[]'ed, memory usage reported by pmap:
 total           419540K
stack memory allocated, memory usage reported by pmap:
 total           419540K
heap memory allocated and delete[]'ed, memory usage reported by pmap:
 total           423708K
*/