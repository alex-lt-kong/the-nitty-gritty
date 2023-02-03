/*
If %module is followed by libs, the shared object must be named "_libs.so"
*/
%module(directors="1") mylibs

%feature("director") MyIf;
%{
/* Put header files here or function declarations like below */
#define SWIG_FILE_WITH_INIT
#include "mylibs.h"
%}


%include "stdint.i"
%include "mylibs.h"
%extend Transcript {
  double __getitem__(size_t i) {
    return $self->Scores[i];
  }
}