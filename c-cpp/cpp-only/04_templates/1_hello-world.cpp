#include <iostream>

using namespace std;

template<typename T>
T my_max(T a, T b) {
    return a > b ? a : b;
}

int main() {
    int max_int;
    double max_dbl;

    max_int = my_max(1, 3);
    cout << max_int << endl;

    max_dbl = my_max(1.0001, 1.0002);
    cout << max_dbl << endl;

    const char* chr_a = "Hello";
    const char* chr_b = "World!";
    cout << my_max(chr_a, chr_b) << endl;
    
    string a = "Hello";
    string b = "World!";
    cout << my_max(a, b) << endl;
    return 0;
}