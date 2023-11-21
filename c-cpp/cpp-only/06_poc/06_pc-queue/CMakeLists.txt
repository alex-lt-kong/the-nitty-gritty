cmake_minimum_required(VERSION 3.12)

project("pc-queue")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD 20)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler-flags.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers.cmake)

add_subdirectory(./src)
