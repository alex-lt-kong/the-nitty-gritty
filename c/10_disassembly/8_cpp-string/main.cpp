#include <string>
#include <iostream>

int main(void) {
    std::string myStr = "This is a comparatively long test string!";
    // To avoid the fuss of templates/constructors/overloading/etc, we
    // use printf() instead of cout in this test.
    printf("%s, size: %lu, capacity: %lu\n",
        myStr.c_str(), myStr.size(), myStr.capacity());
    std::string line;
    std::getline(std::cin, line);
    myStr += line;
    printf("%s, size: %lu, capacity: %lu\n",
        myStr.c_str(), myStr.size(), myStr.capacity());
    return 0;
}