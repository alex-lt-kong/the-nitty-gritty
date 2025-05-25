cmake_minimum_required(VERSION 3.10)

project(manual-memory-management)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(1_usage 1_usage.cpp)
target_link_libraries(1_usage GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(1_usage)

add_executable(2_test-my-new-and-delete 2_my-class-scoped-new-and-delete-test.cpp)
target_link_libraries(2_test-my-new-and-delete GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(2_test-my-new-and-delete)
