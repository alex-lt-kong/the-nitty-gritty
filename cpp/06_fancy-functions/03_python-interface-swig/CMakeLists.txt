# CMakeList.txt : CMake project for python-callback, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("python-interface-swig")

include_directories("W:\\python3.11\\include")
link_directories("W:\\python3.11\\libs")

# Add source to this project's executable.
add_library (libs "libs.cpp")
add_library (libs_wrap "libs_wrap.cxx")
add_library (_libs SHARED "libs.cpp" "libs_wrap.cxx")
