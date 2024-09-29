#include "2_forward_iterator.h"

#include <cstddef>
#include <fmt/core.h>
#include <iostream>
#include <iterator>

int main() {
  IntegersCollection<15> intCollection;
  // statically check whether intCollection has a proper forward_iterator
  // https://en.cppreference.com/w/cpp/iterator/forward_iterator
  static_assert(std::forward_iterator<IntegersCollection<1>::iterator>);
  // Check whether intCollection has a proper forward_iterator at runtime:
  // https://learn.microsoft.com/en-us/cpp/standard-library/iterator-concepts?view=msvc-170#forward_iterator
  fmt::print("intCollection has an iterator: {}\n",
             std::forward_iterator<decltype(intCollection)::iterator>);
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

  GenericCollection<double, 10> dblCollection;
  static_assert(std::forward_iterator<decltype(dblCollection)::iterator>);
  fmt::print("dblCollection has an iterator: {}\n",
             std::forward_iterator<decltype(dblCollection)::iterator>);
  std::iterator_traits<decltype(dblCollection.begin())>::value_type value = 1;
  for (auto itr = dblCollection.begin(); itr != dblCollection.end(); ++itr) {
    value *= 3.1415;
    *itr = value;
    (value) = (long)value % 1024;
  }
  for (auto ele : dblCollection) {
    fmt::print("{:.2f}, ", ele);
  }
  fmt::print("\n");
  return 0;
}