cmake_minimum_required(VERSION 3.13)

project(4_vector-ln LANGUAGES C)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

if(MSVC)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
	set(CMAKE_C_FLAGS "-Wall -Wextra")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
else()
    message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

add_library(
    func
    SHARED
    func.c
 "func.h")
target_link_libraries(
  func
  #PUBLIC
  MKL::MKL
)