#include <iostream>

using namespace std;

void func1(const int& a) {
    cout << a << endl;
    // a = 5; // error: assignment of read-only reference ‘a’
}

void func2(int&& a) {
    cout << a << endl;
}

void func3(int a) {
    cout << a << endl;
}

int main(void) {
    int a = 5;
    return 0;
}