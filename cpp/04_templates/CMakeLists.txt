cmake_minimum_required(VERSION 3.30)
project(ringbuffer)

set(CMAKE_CXX_STANDARD 20)

#find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
#include_directories(${GTEST_INCLUDE_DIRS})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

add_executable(4_perfect-forwarding 4_perfect-forwarding.cpp)
target_link_libraries(4_perfect-forwarding fmt)