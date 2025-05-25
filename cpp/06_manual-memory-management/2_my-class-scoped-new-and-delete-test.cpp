#include "2_my-class-scoped-new-and-delete-impl.h"

#include <gtest/gtest.h>


TEST(TestMyClassScopedNewAndDelete, NewAndDelete) {
    auto o1 = MyClass();
    EXPECT_EQ(o1.m_new_delete_calls_diff, 0);
    EXPECT_EQ(MyClass::m_new_delete_array_calls_diff, 0);
    // It is now in new where we call the constructor, it is MyClass() that
    // invokes the constructor.
    const MyClass *p1 = new MyClass();
    EXPECT_EQ(o1.m_new_delete_calls_diff, 1);
    delete p1;
    EXPECT_EQ(o1.m_new_delete_calls_diff, 0);
    EXPECT_EQ(MyClass::m_new_delete_calls_diff, 0);
    EXPECT_EQ(MyClass::m_new_delete_array_calls_diff, 0);
}


TEST(TestMyClassScopedNewAndDelete, NewArrayAndDeleteArray) {

    const MyClass arr[5] = {3, 1, 4, 1, 5};
    EXPECT_EQ(MyClass::m_new_delete_calls_diff, 0);
    EXPECT_EQ(MyClass::m_new_delete_array_calls_diff, 0);
    EXPECT_EQ(arr[2].m_data, 4);
    EXPECT_EQ(arr[4].m_data, 5);

    const MyClass *ptr = new MyClass[5]{3, 1, 4, 1, 5};
    EXPECT_EQ(MyClass::m_new_delete_calls_diff, 0);
    EXPECT_EQ(MyClass::m_new_delete_array_calls_diff, 1);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(ptr[i].m_data, arr[i].m_data);
    }
    delete[] ptr;
    EXPECT_EQ(MyClass::m_new_delete_array_calls_diff, 0);
    EXPECT_EQ(MyClass::m_new_delete_calls_diff, 0);
}
