cmake_minimum_required(VERSION 3.10)
project(synchronization-demo)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
add_compile_options(-Wall -Wextra)

add_executable(thread-pinning thread-pinning.c)
target_link_libraries(thread-pinning PRIVATE rt pthread)

add_executable(process-pinning process-pinning.c)
target_link_libraries(process-pinning PRIVATE rt pthread)
