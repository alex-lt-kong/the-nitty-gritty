#include <string>
#include <iostream>

using namespace std;

int main() {
    string s0 = to_string(1);
    cout << "s0: " << s0 << endl;
    string s1 = to_string(1) + to_string(2);
    cout << "s1: " << s1 << endl;
    return 0;
}