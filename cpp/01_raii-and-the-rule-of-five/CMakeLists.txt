cmake_minimum_required(VERSION 3.30)
project(01_raii-and-the-rule-of-five)

set(CMAKE_CXX_STANDARD 23)

find_package(GTest CONFIG REQUIRED COMPONENTS GTest GMock)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

add_subdirectory(./rule-of-five)