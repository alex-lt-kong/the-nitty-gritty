cmake_minimum_required(VERSION 3.10)


project(polymorphism)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(driver driver.cpp function-overriding-with-crtp.h)
target_compile_options(driver PRIVATE -O1)

add_executable(function-overloading function-overloading.cpp function-overloading.h)
target_compile_options(driver PRIVATE -O1)