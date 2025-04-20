#include "function-overloading.h"
#include "function-overriding-with-crtp.h"
#include "function-overriding.h"

int main() {
  // Function overloading, no vtable needed as everything is fixed at compile
  // time
  my_print(3);
  my_print(3, 4);
  my_print(3, 4, "5", "Hello world", 3.1415);

  // Function overriding, vtable needed as everything is resolved at runtime
  Base baseObj;
  Derived derivedObj = {};
  show(baseObj);
  show(derivedObj);
  showWithoutVirtual(baseObj);
  showWithoutVirtual(derivedObj);

  // Function overriding with CRTP, no vtable needed as everything is fixed at
  // compile time
  CtrpDummyDerived ctrpBaseObj = {};
  CtrpDerived ctrpDerivedObj = {};
  show(ctrpBaseObj);
  show(ctrpDerivedObj);

  return 0;
}
