cmake_minimum_required(VERSION 3.30)
project(template-implementation)

find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(type-traits-test type-traits-test.cpp)
target_link_libraries(type-traits-test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(type-traits-test)