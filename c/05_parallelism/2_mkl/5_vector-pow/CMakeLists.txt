cmake_minimum_required(VERSION 3.13)

project(4_vector-ln LANGUAGES C)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

message("CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")
if(CMAKE_C_COMPILER_ID MATCHES "MSVC")
	set(CMAKE_C_FLAGS "/O2")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
	set(CMAKE_C_FLAGS "-Wall -Wextra -O3 -Ofast")
else()
    message(WARNING "Unknown CMAKE_C_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
endif()

add_library(
    func
    SHARED
    func.c
)
target_link_libraries(
  func
  #PUBLIC
  MKL::MKL
)