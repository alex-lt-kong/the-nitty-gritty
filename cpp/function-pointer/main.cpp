#include <iostream>
#include <stdio.h>

using namespace std;

int add(int a, int b)
{
    return (a + b);
}
 
int main()
{
    int (*myAdd)(int, int);
    myAdd = add;
    printf("Address of add(): %p\n", &add);
    printf("Value of myAdd:   %p\n", myAdd);
    
    cout << myAdd(1, 2) << endl;
    return 0;
}