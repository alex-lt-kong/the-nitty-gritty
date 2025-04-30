cmake_minimum_required(VERSION 3.30)
project(08_ref_and_pointers)

set(CMAKE_CXX_STANDARD 23)

find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

add_executable(unique_ptr_demo 2_unique-ptr-demo.cpp)
target_link_libraries(unique_ptr_demo GTest::gtest GTest::gtest_main GTest::gmock)

add_executable(2_unique-ptr-impl-test 2_unique-ptr-impl-test.cpp)
target_link_libraries(2_unique-ptr-impl-test GTest::gtest GTest::gtest_main GTest::gmock)

add_executable(1_value-ref-and-pointer 1_value-ref-and-pointer.cpp)
add_executable(observer-pattern 3_observer-pattern.cpp)

add_executable(3_pcqueue-pattern 3_pcqueue-pattern.cpp)

add_executable(3_shared-ptr-demo 3_shared-ptr-demo.cpp)
target_link_libraries(3_shared-ptr-demo GTest::gtest GTest::gtest_main GTest::gmock)