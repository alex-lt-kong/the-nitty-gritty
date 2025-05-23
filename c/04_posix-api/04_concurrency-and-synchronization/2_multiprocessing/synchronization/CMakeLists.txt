cmake_minimum_required(VERSION 3.10)
project(synchronization-demo)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
add_compile_options(-Wall -Wextra)

add_executable(semaphore-daemon semaphore-daemon.c)
target_link_libraries(semaphore-daemon PRIVATE rt pthread)

add_executable(semaphore-on-demand semaphore-on-demand.c)
target_link_libraries(semaphore-on-demand PRIVATE rt pthread)

add_executable(pthread-mutex-in-shm-daemon pthread-mutex-in-shm-daemon.c)
target_link_libraries(pthread-mutex-in-shm-daemon PRIVATE rt pthread)

add_executable(pthread-mutex-in-shm-on-demand pthread-mutex-in-shm-on-demand.c)
target_link_libraries(pthread-mutex-in-shm-on-demand PRIVATE rt pthread)

