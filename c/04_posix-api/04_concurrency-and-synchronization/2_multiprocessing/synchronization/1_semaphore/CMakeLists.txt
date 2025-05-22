cmake_minimum_required(VERSION 3.10)
project(semaphore C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra)

add_executable(semaphore-proc1 semaphore-proc1.c)
target_link_libraries(semaphore-proc1 PRIVATE rt pthread)

add_executable(semaphore-proc2 semaphore-proc2.c)
target_link_libraries(semaphore-proc2 PRIVATE rt pthread)

