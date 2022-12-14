#include <stdio.h>
#include <vector>

int main() {
    std::vector<int> test_vec = {3,1,4,1,5};
    for (size_t i = 0; i < test_vec.size(); ++i) {
        printf("%d\n", test_vec[i]);
    }

    printf("%p\n%p\n%p\n", &test_vec[0], test_vec.data(), test_vec.begin());
}