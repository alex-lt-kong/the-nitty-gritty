#include <memory>
#include <iostream>

using namespace std;

void dumb_func() {
    int* valuePtr = new int(15);
    int x = 45;

    if (x == 45)
        cout << "dumb_func() returned" << endl;
        return;   // here we have a memory leak, valuePtr is not deleted

    delete valuePtr;
}

void smart_func()
{
    std::unique_ptr<int> valuePtr(new int(15));
    int x = 45;

    if (x == 45)
        cout << "smart_func() returned" << endl;
        return;   // no memory leak anymore!

}

int main() {
    dumb_func();
    smart_func();
    return 0;
}