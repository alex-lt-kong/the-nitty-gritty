cmake_minimum_required(VERSION 3.30)
project(08_ref_and_pointers)

set(CMAKE_CXX_STANDARD 23)
#find_package(GTest REQUIRED)
find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(08_ref_and_pointers unique_ptr_demo.cpp)
target_link_libraries(08_ref_and_pointers GTest::gtest GTest::gtest_main GTest::gmock)

add_executable(value-ref-and-pointer value-ref-and-pointer.cpp)
