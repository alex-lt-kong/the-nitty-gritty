#include <iostream>
#include <vector>

int main() {
  std::vector<bool> arr = {true, true, false};

  bool bool_val = arr[1];
  std::cout << "var: " << bool_val << "\n";
  // For space-optimization reasons, the C++ standard (as far back as C++98)
  // explicitly calls out vector<bool> as a special standard container where
  // each bool uses only one bit of space rather than one byte as a normal bool
  // would (implementing a kind of "dynamic bitset"). In exchange for this
  // optimization it doesn't offer all the capabilities and interface of a
  // normal standard container.
  // The below won't compile
  // bool *bool_ptr = &arr[1];
  // auto bool_ref = &arr[1];
  for (const auto &ele : arr) {
    std::cout << ele << ", ";
  }
  std::cout << std::endl;
  return 0;
}