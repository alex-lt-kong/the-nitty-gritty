#include "2_unique-ptr-impl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <print>
#include <stdlib.h>
#include <string.h>

using namespace std;

class MyBoringClass {
public:
  int m_data1;
  double m_data2;
  string m_data3;

  MyBoringClass(const int data1, const double data2, const string &data3)
    : m_data1(data1), m_data2(data2), m_data3(data3) {
  }
};

TEST(MyUniquePtrTest, ConstructorAndDereference) {
  unique_ptr<int> uptr(new int(42));
  my_unique_ptr<int> muptr(new int(42));
  ASSERT_TRUE(uptr != nullptr);
  ASSERT_TRUE(muptr != nullptr);
  EXPECT_EQ(*uptr, 42); // Check dereferenced value
  EXPECT_EQ(*muptr, 42); // Check dereferenced value
}

TEST(MyUniquePtrTest, UniquePtrDoesNotLeak) {
  int *rptr0;
  int *rptr1;
  constexpr int arr_size = 32767;
  auto deleter = [](void *p) { delete[] static_cast<int *>(p); }; {
    // The T vs T[] is more a template programming issue, do not want to get too
    // involved in this section
    unique_ptr<int, decltype(deleter)> uptr(new int[arr_size], deleter);
    my_unique_ptr<int> muptr(new int[arr_size], deleter);
    for (int i = 0; i < arr_size; ++i) {
      uptr.get()[i] = i;
      muptr.get()[i] = i;
    }
    rptr0 = uptr.get();
    rptr1 = muptr.get();
    for (int i = 0; i < arr_size; ++i) {
      EXPECT_EQ(*(uptr.get() + i), i);
      EXPECT_EQ(*(muptr.get() + i), i);
      EXPECT_EQ(*(rptr0 + i), i);
      EXPECT_EQ(*(rptr1 + i), i);
    }
  }

  ASSERT_DEATH(
    {
    for (int i = 0; i < arr_size; i++) {
    EXPECT_EQ(*(rptr0 + i), i);
    }
    },
    ".*");
  ASSERT_DEATH(
    {
    for (int i = 0; i < arr_size; i++) {

    EXPECT_EQ(*(rptr1 + i), i);
    }
    },
    ".*");
}

TEST(MyUniquePtrTest, MoveConstructor) {
  constexpr char text[] = "0xdeadbeef";
  unique_ptr uptr1 = std::make_unique<string>(text);
  my_unique_ptr muptr1 = my_make_unique<string>(text);
  EXPECT_EQ(strcmp(uptr1->data(), text), 0);
  EXPECT_EQ(strcmp(muptr1->data(), text), 0);

  unique_ptr<string> uptr2(std::move(uptr1));
  my_unique_ptr<string> muptr2(std::move(muptr1));
  EXPECT_EQ(strcmp(uptr2->data(), text), 0);
  EXPECT_EQ(strcmp(muptr2->data(), text), 0);
}

TEST(MyUniquePtrTest, MoveAssignmentOperator) {
  constexpr char text[] = "0xdeadbeef";
  unique_ptr<string> uptr1(new string(text));
  my_unique_ptr<string> muptr1(new string(text));
  EXPECT_EQ(strcmp(uptr1->data(), text), 0);
  EXPECT_EQ(strcmp(muptr1->data(), text), 0);

  unique_ptr<string> uptr3;
  my_unique_ptr<string> muptr3;
  ASSERT_TRUE(uptr3 == nullptr);
  ASSERT_TRUE(muptr3 == nullptr);

  uptr3 = std::move(uptr1);
  muptr3 = std::move(muptr1);

  ASSERT_TRUE(uptr1 == nullptr);
  ASSERT_TRUE(muptr1 == nullptr);
  EXPECT_EQ(strcmp(uptr3->data(), text), 0);
  EXPECT_EQ(strcmp(muptr3->data(), text), 0);

  uptr3 = std::move(uptr3);
  muptr3 = std::move(muptr3);
  EXPECT_EQ(strcmp(uptr3->data(), text), 0);
  EXPECT_EQ(strcmp(muptr3->data(), text), 0);
}

TEST(MyUniquePtrTest, Swap) {
  const auto obj1 = MyBoringClass(1, 3.14, "HelloWorld");
  const auto obj2 = MyBoringClass(65535, 2.71, "0x1234");
  auto uptr1 = std::make_unique<MyBoringClass>(obj1);
  auto uptr2 = std::make_unique<MyBoringClass>(obj2);
  uptr1.swap(uptr2);
  EXPECT_EQ(uptr1->m_data1, 65535);
  EXPECT_EQ(uptr2->m_data2, 3.14);
  auto muptr1 = my_make_unique<MyBoringClass>(obj1);
  auto muptr2 = my_make_unique<MyBoringClass>(obj2);
  muptr1.swap(muptr2);
  EXPECT_EQ(muptr1->m_data3, "0x1234");
  EXPECT_EQ(muptr2->m_data3, "HelloWorld");
}

class MockHelper {
public:
  MOCK_METHOD(void, Call, ());
};

TEST(MyUniquePtrTest, UniquePtrFromRawPtr) {
  int arr_size = 65546;
  MockHelper mockHelper;
  auto deleter = [&mockHelper](int *ptr) {
    std::free(ptr);
    mockHelper.Call();
  };
  auto raw_ptr1 = static_cast<int *>(malloc(sizeof(int) * arr_size));
  for (int i = 0; i < arr_size; i++) {
    raw_ptr1[i] = i;
  }
  auto raw_ptr2 = static_cast<int *>(malloc(sizeof(int) * arr_size));

  EXPECT_CALL(mockHelper, Call()).Times(2);
  memcpy(raw_ptr2, raw_ptr1, sizeof(int) * arr_size); {
    unique_ptr<int, decltype(deleter)> uptr(raw_ptr1, deleter);
    my_unique_ptr<int> muptr(raw_ptr2, deleter);
    EXPECT_EQ(uptr.get(), raw_ptr1);
    EXPECT_EQ(muptr.get(), raw_ptr2);

    for (size_t i = 0; i < arr_size; ++i) {
      EXPECT_EQ(uptr.get()[i], i);
      EXPECT_EQ(muptr.get()[i], i);
    }
  }
}
