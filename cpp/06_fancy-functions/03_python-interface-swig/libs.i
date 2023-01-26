/*
If %module is followed by libs, the shared object must be named "_libs.so"
*/
%module libs
%{
/* Put header files here or function declarations like below */

#include "libs.h"
%}
%include "libs.h"
%include "std_string.i"

%extend Transcript {
  double __getitem__(size_t i) {
    return $self->Scores[i];
  }
}