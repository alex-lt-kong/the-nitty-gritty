cmake_minimum_required(VERSION 3.16.3)

project(CAMERA_SERVER)

set (CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -pedantic -O3 -Wno-c11-extensions)


add_subdirectory(src/camera-server)
