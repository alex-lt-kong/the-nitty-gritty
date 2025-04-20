cmake_minimum_required(VERSION 3.10)


project(driver)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(driver driver.cpp
        function-overriding-with-crtp.h)
target_include_directories(driver PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(driver PRIVATE -O1)