#include <gmock/gmock.h>
#include <gtest/gtest.h>

// #include <iostream>
#include <memory>
#include <print>

using namespace std;

TEST(UniquePtrTest, ConstructorAndDereference) {
  unique_ptr<int> uptr(new int(42));
  ASSERT_TRUE(uptr);    // Check that the pointer is not null
  EXPECT_EQ(*uptr, 42); // Check dereferenced value
}

TEST(UniquePtrTest, DumbPtrLeaks) {
  int *valuePtr = new int(15);
  int x = 45;
  if (x == 45) {
    return; // here we have a memory leak, valuePtr is not deleted
  }
  delete valuePtr;
}

TEST(UniquePtrTest, UniquePtrDoesNotLeak) {
  unique_ptr<int> valuePtr(new int(15));
  int x = 45;
  if (x == 45) {
    // no memory leak anymore as unique_ptr's destructor is called to release
    // the recource.
    return;
  }
}

TEST(UniquePtrTest, MoveConstructor) {
  unique_ptr<int> uptr1(new int(123));
  unique_ptr<int> uptr2(std::move(uptr1)); // Move ownership
  ASSERT_FALSE(uptr1);                     // Original pointer should be null
  ASSERT_TRUE(uptr2); // New pointer should own the resource
  EXPECT_EQ(*uptr2, 123);
}

class MockHelper {
public:
  MOCK_METHOD(void, Call, ());
};

TEST(UniquePtrTest, UniquePtrFromRawPtr) {
  int arr_size = 32767;
  MockHelper mockHelper;
  auto deleter = [&mockHelper](int *ptr) {
    std::free(ptr);
    mockHelper.Call();
  };
  auto raw_ptr = (int *)malloc(sizeof(int) * arr_size);
  for (int i = 0; i < arr_size; i++) {
    raw_ptr[i] = i;
  }
  {
    unique_ptr<int[], decltype(deleter)> smart_int_ptr(raw_ptr, deleter);

    for (size_t i = 0; i < arr_size; ++i) {
      EXPECT_EQ(smart_int_ptr[i], i);
    }

    EXPECT_CALL(mockHelper, Call()).Times(1);
  }
  // An indirect way to check the memory is released.
  ASSERT_DEATH(
      {
        for (int i = 0; i < arr_size; i++) {
          std::println("{}", raw_ptr[i]);
        }
      },
      ".*"); // The program should crash here, as the memory is already
             // released.

  // the below check wont work as raw_arr will still point to the previous
  // location, even though the location is no long valid.
  // EXPECT_EQ(*dynamic_int_arr, nullptr);
}

void callee_func_raw_ptr(int *arg) { ++(*arg); }

void callee_func_ref(unique_ptr<int> &arg) { ++(*arg); }

void callee_func_move(unique_ptr<int> arg) { ++(*arg); }

TEST(UniquePtrTest, PassUniquePtrToFunctions) {

  constexpr int val = 45;
  unique_ptr<int> x(new int(val));
  callee_func_ref(x);
  EXPECT_EQ(*x, val + 1);
  callee_func_raw_ptr(x.get());
  EXPECT_EQ(*x, val + 2);
  callee_func_move(move(x));
  EXPECT_FALSE(x.get());
}

unique_ptr<int> get_ptr() {
  unique_ptr<int> x(new int(31415));
  return x;
}

TEST(UniquePtrTest, ReturnUniquePtrFromFunc) {
  auto x = get_ptr();
  EXPECT_EQ(*x, 31415);
}