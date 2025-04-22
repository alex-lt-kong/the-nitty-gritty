cmake_minimum_required(VERSION 3.10)


project(polymorphism)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(crtp crtp.cpp)
target_compile_options(crtp PRIVATE -O1)

add_executable(function-overriding function-overriding.cpp)
target_compile_options(function-overriding PRIVATE -O1)

add_executable(function-overloading function-overloading.cpp)
target_compile_options(function-overloading PRIVATE -O1)