#include <gtest/gtest.h>


TEST(ManualMemoryManagementTest, TestNewAndDelete) {
    {
        int *ptr = new int;
        *ptr = 42;
        EXPECT_EQ(*ptr, 42);
        delete ptr;
    }
    {
        int *arr = new int[5]{3, 1, 4, 1, 5};
        EXPECT_EQ(arr[1], 1);
        EXPECT_EQ(arr[2], 4);

        delete[] arr;
    }
}

TEST(ManualMemoryManagementTest, TestStdAllocator) {
    // here we dont need to call constructor/destructor only because int is a
    // "primitive type". If we had a class, we would need to call the
    // constructor and destructor manually.
    {
        std::allocator<int> alloc;
        int *ptr = alloc.allocate(1);
        *ptr = 43;
        EXPECT_EQ(*ptr, 43);
        alloc.deallocate(ptr, 1);
    }
    {
        std::allocator<int> alloc;
        constexpr size_t sz = 5;
        int *ptr = alloc.allocate(sz);
        ptr[0] = 3;
        ptr[1] = 1;
        ptr[2] = 4;
        ptr[3] = 1;
        ptr[4] = 5;
        EXPECT_EQ(ptr[1], 1);
        EXPECT_EQ(ptr[2], 4);
        alloc.deallocate(ptr, sz);
    }
}

class MyClass {
public:
    int m_data = -1;
    MyClass() { m_data = 42; }
};

TEST(ManualMemoryManagementTest, NewCallsConstructorButMallocDoesNot) {
    {
        auto *ptr = new MyClass();
        EXPECT_EQ(ptr->m_data, 42);
        delete ptr;
    }
    {
        auto *raw_ptr = static_cast<MyClass *>(malloc(sizeof(MyClass)));
        EXPECT_NE(raw_ptr->m_data, 42);
        // This feature is called "placement new"
        const auto ptr = new (raw_ptr) MyClass();
        EXPECT_EQ(ptr->m_data, 42);
        ptr->~MyClass();
        free(ptr);
        // delete raw_ptr;
    }
    {
        auto *ptr = new (std::malloc(sizeof(MyClass))) MyClass();
        EXPECT_EQ(ptr->m_data, 42);
        ptr->~MyClass();
        free(ptr);
        // Note: We should not use delete here, as it does not call the
        // destructor delete ptr;
    }
}


TEST(ManualMemoryManagementTest, StdAllocatorDoesNotCallConstructorEither) {

    std::allocator<MyClass> alloc;
    MyClass *ptr = alloc.allocate(1);
    EXPECT_NE(ptr->m_data, 42);
    // Interally calls a fancy version of placement new, but available since
    // C++20 https://en.cppreference.com/w/cpp/memory/construct_at
    std::construct_at(ptr);
    EXPECT_EQ(ptr->m_data, 42);
    std::destroy_at(ptr);
    alloc.deallocate(ptr, 1);
}
