# CMakeList.txt : CMake project for python-callback, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("python-callback")

# Add source to this project's executable.
add_library (python-callback SHARED "func.c" "func.h")

# TODO: Add tests and install targets if needed.
