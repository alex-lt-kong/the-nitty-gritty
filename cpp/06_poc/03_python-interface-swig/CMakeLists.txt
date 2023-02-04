cmake_minimum_required(VERSION 3.16)

project(MylibsWrapperProj VERSION 1.0 LANGUAGES CXX)
add_compile_options(-Wall -Wextra -pedantic -O3)
if ( MSVC )
  set(SWIG_DIR "C:/tmp/swig-repo/swig/Lib")
  set(SWIG_EXECUTABLE "C:/tmp/swigwin-4.1.1/swig.exe")  
endif (MSVC)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})


# Set .Net project directory
set(NET_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csharp-wrapper/csharp-wrapper")

# Add swig flags here
set(CMAKE_SWIG_FLAGS "")


set_property(SOURCE mylibs.i PROPERTY CPLUSPLUS ON)
set_source_files_properties(mylibs.i PROPERTIES SWIG_FLAGS "-includeall")

swig_add_library(mylibs
  TYPE SHARED
  LANGUAGE CSharp
  SOURCES mylibs.i mylibs.h
  OUTPUT_DIR ${NET_PROJECT_DIR}
  OUTFILE_DIR ${NET_PROJECT_DIR}
  )

# for copying mylibs.dll to .Net project dir dir
set_target_properties( mylibs
    PROPERTIES
    # These copy animal.dll on Windows to .Net project directory
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${NET_PROJECT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${NET_PROJECT_DIR}
    
    # This copies mylibs.so on Linux to .Net project directory
    LIBRARY_OUTPUT_DIRECTORY ${NET_PROJECT_DIR}

    # Set address of C++ headers
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
)