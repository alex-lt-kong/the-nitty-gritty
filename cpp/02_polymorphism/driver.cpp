#include "function-overriding-with-crtp.h"
#include "function-overriding.h"

int main() {

  // Function overriding, vtable needed as everything is resolved at runtime
  Base baseObj;
  Derived derivedObj = {};
  show(baseObj);
  show(derivedObj);
  showWithoutVirtual(baseObj);
  showWithoutVirtual(derivedObj);

  // Function overriding with CRTP, no vtable needed as everything is fixed at
  // compile time
  CrtpDummyDerived crtpBaseObj = {};
  CrtpDerived crtpDerivedObj = {};
  show(crtpBaseObj);
  show(crtpDerivedObj);

  return 0;
}
