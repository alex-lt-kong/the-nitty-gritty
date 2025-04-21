#include "function-overriding.h"

int main() {

  // Function overriding, vtable needed as everything is resolved at runtime
  Base baseObj;
  Derived derivedObj = {};
  show(baseObj);
  show(derivedObj);
  showWithoutVirtual(baseObj);
  showWithoutVirtual(derivedObj);

  return 0;
}
