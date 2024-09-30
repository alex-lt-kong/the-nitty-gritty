// At a high level, a range is something that you can iterate over. A range is
// represented by an iterator that marks the beginning of the range and a
// sentinel that marks the end of the range.
// https://learn.microsoft.com/en-us/cpp/standard-library/ranges?view=msvc-170

// The C++ Standard Library containers such as vector and list are ranges
// https://devblogs.microsoft.com/cppblog/documentation-for-cpp20-ranges/
#include <fmt/format.h>

#include <algorithm>
#include <ranges>
#include <vector>

void no_range(const std::vector<int> input) {
  std::vector<int> intermediate, output;

  std::copy_if(input.begin(), input.end(), std::back_inserter(intermediate),
               [](const int i) { return i % 3 == 0; });
  std::transform(intermediate.begin(), intermediate.end(),
                 std::back_inserter(output), [](const int i) { return i * i; });
  fmt::print("output: {}\n", fmt::join(output, ", "));
}

void yes_range(const std::vector<int> input) {
  auto output = input |
                std::views::filter([](const int n) { return n % 3 == 0; }) |
                std::views::transform([](const int n) { return n * n; });
  fmt::print("output: {}\n", fmt::join(output, ", "));
}

int main() {
  std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // Before ranges, if you wanted to transform the elements of a collection that
  // met a certain criterion, you needed to introduce an intermediate step to
  // hold the results between operations.
  fmt::print("input: {}\n", fmt::join(input, ", "));
  // For example, if you wanted to build a vector of squares from the elements
  // in another vector that are divisible by three, you could write something
  // like no_range():
  no_range(input);
  // With ranges, you can accomplish the same thing without needing the
  // intermediate vector:
  yes_range(input);
  return 0;
}