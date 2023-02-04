/*
If %module is followed by libs, the shared object must be named "_libs.so"
*/
%module(directors="1") mylib

%feature("director");
%{
/* Put header files here or function declarations like below */
#define SWIG_FILE_WITH_INIT
#include "mylib.h"
%}


%include "stdint.i"
%include "mylib.h"
