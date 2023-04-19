#include <vector>
#include <iostream>
#include <ostream>

using namespace std;

int main(void) {
    vector<int> vec = {1, 2, 3, 4, 5};
    cout << "Read-only iteration by ref: ";
    /* Using reference is very important as for more complicated objects
    accessing them by value means their copy constructors must be called */
    for (const int& ele : vec) {
        cout << ele << ", ";
    }
    cout << endl;

    cout << "Read-only iteration by ref with auto: ";
    /* If the type is well-known, such as `vector<int> vec;`, or no
    one really cares, such as `auto add = [&](int a, int b) { return a+b };`
    using auto may save us some time.*/
    for (const auto& ele : vec) {
        cout << ele << ", ";
    }
    cout << endl;

    cout << "Read-only iteration by value (if objects are cheap to copy): ";
    for (const auto ele : vec) {
        cout << ele << ", ";
    }
    cout << endl;

    cout << "Read-write iteration, has to be by ref: ";
    for (auto& ele : vec) {
        cout << ++ele << ", ";
    }
    cout << endl;

    cout << "Read-write iteration: ";
    for (auto& ele : vec) {
        cout << ele << ", ";
    }
    cout << endl;
    return 0;
}