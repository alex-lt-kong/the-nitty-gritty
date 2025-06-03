cmake_minimum_required(VERSION 3.20)
project(template-implementation)

find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(6_building-blocks-of-metaprogramming-test 6_building-blocks-of-metaprogramming-test.cpp)
target_link_libraries(6_building-blocks-of-metaprogramming-test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(6_building-blocks-of-metaprogramming-test)

add_executable(7_type-traits-test 7_type-traits-test.cpp)
target_link_libraries(7_type-traits-test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(7_type-traits-test)