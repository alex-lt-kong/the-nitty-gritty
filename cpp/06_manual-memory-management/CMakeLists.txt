cmake_minimum_required(VERSION 3.10)

project(manual-memory-management)

find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(1_usage 1_usage.cpp)
target_link_libraries(1_usage GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(1_usage)
