cmake_minimum_required(VERSION 3.12)
project(2_iterator)

set(CMAKE_CXX_STANDARD 23)

#find_package(fmt CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(1_usage 1_usage.cpp)
target_link_libraries(1_usage GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(1_usage)

add_executable(2_forward-iterator-test 2_forward-iterator-test.cpp)
target_link_libraries(2_forward-iterator-test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(2_forward-iterator-test)

add_executable(3_bidirectional-iterator-test 3_bidirectional-iterator-test.cpp)
target_link_libraries(3_bidirectional-iterator-test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(3_bidirectional-iterator-test)

add_executable(4_random-access-iterator-test 4_random-access-iterator-test.cpp)
target_link_libraries(4_random-access-iterator-test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(4_random-access-iterator-test)

