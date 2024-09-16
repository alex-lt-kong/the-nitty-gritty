#include <stdio.h>
#include <vector>
#include <string.h>

int main() {
    std::vector<int> vec = {3,1,4,1,5};
    for (size_t i = 0; i < vec.size(); ++i) {
        printf("%d,", vec[i]);
    }
    printf("\n\n");

    printf("Get a vector's internal C array:\n");
    printf("%p\n%p\n%p\n%p\n%p\n", &vec[0], vec.data(), vec.begin(), &vec.front(), &vec.at(0));
    printf("\n");

    int arr[] = {6,5,5,3,6};
    printf("memcpy() data directly into a vector is okay:\n");
    memcpy(vec.data(), arr, sizeof(arr));
    for (size_t i = 0; i < vec.size(); ++i) {
        printf("%d,", vec[i]);
    }
    printf("\n\n");
}