#include <gtest/gtest.h>

TEST(ManualMemoryManagementTest, TestNewAndDelete) {
    int *ptr = new int;
    *ptr = 42;
    EXPECT_EQ(*ptr, 42);
    delete ptr;

    int *arr = new int[5]{3, 1, 4, 1, 5};
    EXPECT_EQ(arr[1], 1);
    EXPECT_EQ(arr[2], 4);

    delete[] arr;
}
