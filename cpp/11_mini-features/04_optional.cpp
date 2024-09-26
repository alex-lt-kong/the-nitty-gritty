#include <iostream>
#include <optional>

using namespace std;

int my_optional_addition(int a, optional<int> b) {
  if (b) {
    return a + b.value();
  }
  return a;
}

optional<float> my_division(float dividend, float divisor) {
  if (divisor != 0) {
    return dividend / divisor;
  }
  return {};
}

int main() {
  cout << my_optional_addition(2, 4) << endl;
  cout << my_optional_addition(2, nullopt) << endl;

  optional<float> quotient = my_division(32, 3.141);
  if (quotient) {
    cout << quotient.value() << endl;
  } else {
    cerr << "Divided by 0!" << endl;
  }
  quotient = my_division(0, 0);
  if (quotient) {
    cout << quotient.value() << endl;
  } else {
    cerr << "Divided by 0!" << endl;
  }

  return 0;
}