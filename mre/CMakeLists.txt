cmake_minimum_required(VERSION 3.16.3)

project(CAMERA_SERVER)

set (CMAKE_CXX_STANDARD 17)

find_package(nlohmann_json CONFIG REQUIRED)

find_library(SPDLOG_LIB spdlog)
if(NOT SPDLOG_LIB)
    message(
        FATAL_ERROR
        "spdlog library not found, install it with 'apt install libspdlog-dev'"
    )
endif()

add_compile_options(-Wall -Wextra -pedantic -O3 -Wno-c11-extensions)


add_subdirectory(src/camera-server)
