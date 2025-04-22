#include "crtp.h"

int main() {

  // Function overriding with CRTP, no vtable needed as everything is fixed at
  // compile time
  EssentiallyBase baseObj = {};
  Derived derivedObj = {};
  show(baseObj);
  show(derivedObj);

  return 0;
}
