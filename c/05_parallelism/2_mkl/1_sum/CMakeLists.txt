cmake_minimum_required(VERSION 3.13)

project(oneMKL_Example LANGUAGES C)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

# Detect the compiler
if(MSVC)
    # Microsoft Visual C++ compiler
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
	set(CMAKE_C_FLAGS "-Wall -Wextra")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
else()
    message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()


add_executable(main func.c)
target_link_libraries(main PUBLIC MKL::MKL)
add_library(
    func
    SHARED
    func.c
)

target_link_libraries(
    func
  PUBLIC MKL::MKL
)