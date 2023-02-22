#include <string>
#include <iostream>

using namespace std;

int main() {
    string s0 = to_string(12);
    cout << "s0: " << s0 << endl;
    string s1 = to_string(34) + to_string(56);
    cout << "s1: " << s1 << endl;
    return 0;
}