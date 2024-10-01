// ref:
// https://learn.microsoft.com/en-us/cpp/standard-library/ranges?view=msvc-170
// A view is a lightweight range. View operations--such as default construction,
// move construction/assignment, copy construction/assignment (if present),
// destruction, begin, and end--all happen in constant time regardless of the
// number of elements in the view.

#include <fmt/format.h>

// requires /std:c++20
#include <iostream>
#include <ranges>
#include <vector>

void use_views() {
  fmt::print("use_views():\n");
  std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fmt::print("input: {}\n", fmt::join(input, ", "));
  auto divisible_by_three = [](const int n) { return n % 3 == 0; };
  auto square = [](const int n) { return n * n; };
  // Views are "composable", meaning that they can be chained together with
  // pipe (|) symbols to represent a sequence of transformations
  auto x = input | std::views::filter(divisible_by_three) |
           std::views::transform(square);

  fmt::print("x: {}\n", fmt::join(x, ", "));
  fmt::print("\n");
}

void range_adaptors() {
  // Range adaptors create a view from a range.
  fmt::print("range_adaptors():\n");
  std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fmt::print("input:               {}\n", fmt::join(input, ", "));

  auto divisible_by_three = [](const int n) { return n % 3 == 0; };
  // std::views::filter() is a range adaptor
  auto output_div_by_three = input | std::views::filter(divisible_by_three);
  fmt::print("output_div_by_three: {}\n", fmt::join(output_div_by_three, ", "));

  // std::views::reverse is another range adaptor
  auto output_reverse = input | std::views::reverse;
  fmt::print("output_reverse:      {}\n", fmt::join(output_reverse, ", "));

  // std::views::output_take is another range adaptor
  auto output_take = input | std::views::take(5);
  fmt::print("output_take:         {}\n", fmt::join(output_take, ", "));

  auto output_transform =
      input | std::views::transform([](int ele) { return ele * 2; });
  fmt::print("output_transform:    {}\n", fmt::join(output_transform, ", "));

  std::string string_input = "The|quick|brown|fox|jumps|over|the|lazy|dog";
  std::string delim = "|";
  auto output_split = string_input | std::views::split(delim);
  for (const auto &part : output_split) {
    fmt::print("{} ", std::string_view(part.begin(), part.end()));
  }
  fmt::print("\n");
  // fmt::print("output_split:         {}\n", fmt::join(output_split, ", "));
  fmt::print("\n");
}

void range_factories() {
  // Range adaptors create a view from something that is NOT a range.
  fmt::print("range_factories():\n");
  auto numbers_view = std::views::iota(1, 11);
  fmt::print("numbers_view:   {}\n", fmt::join(numbers_view, ", "));
  // We need a vector to materialize the view...
  std::vector<int> numbers_vec(numbers_view.begin(), numbers_view.end());
  fmt::print("numbers_vec:    {}\n", fmt::join(numbers_vec, ", "));

  fmt::print("\n");
}

int main() {
  use_views();
  //  range adaptors (basically, algorithms that take one or more ranges and
  //  return a new “adapted” range) and range factories (algorithms that return
  //  a range but without a range as input)
  // https://brevzin.github.io/c++/2021/02/28/ranges-reference/#join
  range_adaptors();
  range_factories();
  return 0;
}
