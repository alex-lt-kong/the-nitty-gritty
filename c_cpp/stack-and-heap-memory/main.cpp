#include<iostream>
#include<cstring>
#include <sys/resource.h>

using namespace std;


class Test {  

  public:
    int itemCount = 0;

    void PrintProcessStackSize() {
        struct rlimit limit;

        getrlimit(RLIMIT_STACK, &limit);
        // RLIMIT_STACK is the maximum size of the process stack, in bytes.
        // Upon reaching this limit, a SIGSEGV (segmentation violation or 
        // segmentation fault) signal is generated.
        // On Linux, you can also check this information by issuing
        // ulimit -a | grep stack
        cout << "Process Stack's soft limit: " << limit.rlim_cur / 1024
             << " KBytes, hard limit: " << limit.rlim_max / 1024 << " KBytes" << endl;
        this->itemCount = limit.rlim_cur / sizeof(long);
        cout << "Max length of a long array: " << this->itemCount << endl;
        this->itemCount++;
    }

    void StackMemoryAllocation() 
    {
        cout << "About to define: long array[" << this->itemCount << "]" << endl;
        long array[this->itemCount];
        for (int i = 0; i < this->itemCount; i++) {
          array[i] = i;
        }
    }

    void HeapMemoryAllocation() {
        cout << "About to define: long* array = (long*) malloc(sizeof(long) * " << this->itemCount << ")" << endl;
        long* array = (long*) malloc(sizeof(long) * this->itemCount);
        for (int i = 0; i < this->itemCount; i++) {
          array[i] = i;
        }
    }
};

int main(int argc, char *argv[])
{
    
    if (argc < 2) {
      cerr << "No argument is received" << endl;
      return 1;
    } else if (argc > 2) {
      cerr << "More than one argument is received" << endl;
      return 1;
    }
    cout << "sizeof(int): " << sizeof(int) << ", sizeof(long):" << sizeof(long) << endl;
    // the size of int and long may or may not be the same on a machine
    Test* myTest = new Test();
    myTest->PrintProcessStackSize();
    if (strcmp(argv[1], "stack") == 0) {
      myTest->StackMemoryAllocation();
    }
    else if (strcmp(argv[1], "heap") == 0) {
        myTest->HeapMemoryAllocation();
    }
    else {
      cerr << "Valid arguments are 'heap' and 'stack' only" << endl;
      return 1;
    }
    delete myTest;
    
    return 0;
}