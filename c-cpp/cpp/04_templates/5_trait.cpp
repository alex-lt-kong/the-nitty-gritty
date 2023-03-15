#include <iostream>
#include <limits>
#include <limits.h>
#include <float.h>
using namespace std;

template <typename T>
bool greater_than_half_max_without_trait(T val) {
    return false;
}

template <>
bool greater_than_half_max_without_trait<int32_t>(int32_t val) {
    return (val > INT_MAX / 2);
}
template <>
bool greater_than_half_max_without_trait<int64_t>(int64_t val) {
    return (val > INT64_MAX / 2);
}
template <>
bool greater_than_half_max_without_trait<float>(float val) { 
    return (val > FLT_MAX / 2);
}
template <>
bool greater_than_half_max_without_trait<double>(double val) {    
    return  (val > DBL_MAX / 2);
}
template <>
bool greater_than_half_max_without_trait<uint32_t>(uint32_t val) {    
    return  (val > UINT32_MAX / 2);
}

template <typename T>
bool greater_than_half_max_with_trait(T val) {
    return (val > numeric_limits<T>::max() / 2);
}

int main() {
    int a = 2147483647;
    uint32_t b = 5432109;
    double c = 2147483648;
    cout << "greater_than_half_max_without_trait(a): "
         << greater_than_half_max_without_trait(a) << endl;
    cout << "greater_than_half_max_without_trait(b): "
         << greater_than_half_max_without_trait(b) << endl;
    cout << "greater_than_half_max_without_trait(c): "
         << greater_than_half_max_without_trait(c) << endl;

    cout << "greater_than_half_max_with_trait(a): "
         << greater_than_half_max_with_trait(a) << endl;
    cout << "greater_than_half_max_with_trait(b): "
         << greater_than_half_max_with_trait(b) << endl;
    cout << "greater_than_half_max_with_trait(c): "
         << greater_than_half_max_with_trait(c) << endl;
    return 0;
}