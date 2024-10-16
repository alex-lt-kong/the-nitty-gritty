cmake_minimum_required(VERSION 3.13)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(1_sum LANGUAGES C)
find_package(MKL CONFIG REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast")

add_executable(main func.c)


target_compile_options(main PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(main PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(main PUBLIC $<LINK_ONLY:MKL::MKL>)

add_library(
    func
    SHARED
    func.c
)

target_link_libraries(
    func
    PUBLIC $<LINK_ONLY:MKL::MKL>
)