#include "3_my-global-scoped-new-and-delete-impl.h"

#include <gtest/gtest.h>

class MyClass {

public:
    int m_data;

    MyClass() : m_data(0) {}

    MyClass(const int data) : m_data(data) {}
};

TEST(TestMyGlobalScopedNewAndDelete, NewAndDelete) {
    auto o1 = MyClass(23333);
    // For global-scoped new/delete, you can expect it to be 0/1, as it will be used by other components too.

    const MyClass *p1 = new MyClass(23333);
    EXPECT_GE(new_delete_calls_diff, 1);
    // For global-scoped new/delete, you can't expect it to be 0/1, as it will be used by other components too.
    EXPECT_GT(new_delete_calls_diff, 1);
    EXPECT_EQ(o1.m_data, p1->m_data);
    auto diff = new_delete_calls_diff;
    delete p1;
    EXPECT_EQ(new_delete_calls_diff, diff - 1);
}


TEST(TestMyGlobalScopedNewAndDelete, NewArrayAndDeleteArray) {

    const MyClass arr[5] = {3, 1, 4, 1, 5};
    EXPECT_EQ(arr[2].m_data, 4);
    EXPECT_EQ(arr[4].m_data, 5);

    const MyClass *ptr = new MyClass[5]{3, 1, 4, 1, 5};
    EXPECT_GE(new_delete_array_calls_diff, 1);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(ptr[i].m_data, arr[i].m_data);
    }
    auto diff = new_delete_array_calls_diff;
    delete[] ptr;
    EXPECT_EQ(new_delete_array_calls_diff, diff - 1);
}
