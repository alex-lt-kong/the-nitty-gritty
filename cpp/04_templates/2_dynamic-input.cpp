#include <iostream>

using namespace std;

template<typename T>
T my_max(T a, T b) {
    return a > b ? a : b;
}

int main() {
    srand(time(NULL));

    int max_int;
    int a_int = rand(), b_int = rand();
    max_int = my_max(a_int, b_int);
    cout << max_int << endl;
    
    double max_dbl;
    double a_dbl = (double)rand() / rand();
    double b_dbl = (double)rand() / rand();
    max_dbl = my_max(a_dbl, b_dbl);
    cout << max_dbl << endl;
    return 0;
}