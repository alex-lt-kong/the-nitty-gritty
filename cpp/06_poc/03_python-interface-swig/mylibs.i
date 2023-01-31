/*
If %module is followed by libs, the shared object must be named "_libs.so"
*/
%module mylibs
%{
/* Put header files here or function declarations like below */

#include "mylibs.h"
%}
%include "mylibs.h"

%extend Transcript {
  double __getitem__(size_t i) {
    return $self->Scores[i];
  }
}