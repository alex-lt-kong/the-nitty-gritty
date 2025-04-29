cmake_minimum_required(VERSION 3.30)
project(08_ref_and_pointers)

set(CMAKE_CXX_STANDARD 23)

find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

add_executable(unique_ptr_demo unique_ptr_demo.cpp)
target_link_libraries(unique_ptr_demo GTest::gtest GTest::gtest_main GTest::gmock)

add_executable(unique_ptr_impl_test unique_ptr_impl_test.cpp)
target_link_libraries(unique_ptr_impl_test GTest::gtest GTest::gtest_main GTest::gmock)

add_executable(value-ref-and-pointer value-ref-and-pointer.cpp)
add_executable(observer-pattern observer-pattern.cpp)
add_executable(shared_ptr_demo shared_ptr_demo.cpp)
add_executable(pcqueue-pattern pcqueue-pattern.cpp)
