#include <fmt/core.h>
#include <iostream>
#include <vector>

int main() {

  std::vector<std::string> words = {"The",  "quick", "brown", "fox", "jumps",
                                    "over", "the",   "lazy",  "dog"};

  // create an iterator to a string vector
  std::vector<std::string>::iterator itr;

  // iterate over all elements
  for (itr = words.begin(); itr != words.end(); itr++) {
    fmt::print("{}, ", *itr);
  }

  return 0;
}