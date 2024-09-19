#include "2_forward_iterator.h"

#include <cstddef>
#include <fmt/core.h>
#include <iterator>

int main() {
  IntegersCollection intCollection;

  int count = 1;
  for (auto itr = intCollection.begin(); itr != intCollection.end(); ++itr) {
    count *= 3;
    *itr = count;
    count %= 1024;
  }

  for (auto itr = intCollection.begin(); itr != intCollection.end(); ++itr) {
    fmt::print("{}, ", *itr);
  }
  fmt::print("\n");
  for (auto ele : intCollection) {
    fmt::print("{}, ", ele);
  }
  fmt::print("\n");

  GenericCollection<double> dblCollection;
  std::iterator_traits<decltype(dblCollection.begin())>::value_type value = 1;
  for (auto itr = dblCollection.begin(); itr != dblCollection.end(); ++itr) {
    value *= 3.1415;
    *itr = value;
    (value) = (long)value % 1024;
  }
  for (auto ele : dblCollection) {
    fmt::print("{}, ", ele);
  }
  fmt::print("\n");
  return 0;
}