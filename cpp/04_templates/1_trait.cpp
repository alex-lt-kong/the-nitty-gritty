#include <array>
#include <cfloat>
#include <climits>
#include <deque>
#include <iostream>
#include <limits>
#include <list>

/* Think of a trait as a small object whose main purpose is to carry information
 * used by another object or algorithm to determine "policy" or "implementation
 * details". - Bjarne Stroustrup  */

using namespace std;

// Define a concept for an iterable container
template <typename T>
concept Iterable = requires(T &t) {
  std::begin(t);
  std::end(t);
  typename T::value_type;
  { *std::begin(t) } -> std::convertible_to<typename T::value_type>;
};

// Template function that requires an Iterable type
template <Iterable T> typename T::value_type findMax(const T &collection) {
  // std::numeric_limits<double>::min(); is a trait
  typename T::value_type largest =
      std::numeric_limits<typename T::value_type>::min();
  for (const auto &ele : collection) {
    if (ele > largest)
      largest = ele;
  }
  return largest;
}

/*
Now we are going to implement the same trait ourselves
*/
namespace my {

template <typename T> struct numeric_limits;

// Specialization for integer types
template <> struct numeric_limits<int> {
  static constexpr int min() { return INT_MIN; }
};

template <> struct numeric_limits<long> {
  static constexpr long min() { return LONG_MIN; }
};

template <> struct numeric_limits<long long> {
  static constexpr long long min() { return LLONG_MIN; }
};

template <> struct numeric_limits<unsigned int> {
  static constexpr unsigned int min() { return 0; }
};

template <> struct numeric_limits<unsigned long> {
  static constexpr unsigned long min() { return 0; }
};

template <> struct numeric_limits<unsigned long long> {
  static constexpr unsigned long long min() { return 0; }
};

// Specialization for floating-point types
template <> struct numeric_limits<float> {
  static constexpr float min() { return FLT_MIN; }
};

template <> struct numeric_limits<double> {
  static constexpr double min() { return DBL_MIN; }
};

template <> struct numeric_limits<long double> {
  static constexpr long double min() { return LDBL_MIN; }
};
template <> struct numeric_limits<std::string> {
  static constexpr std::string min() { return ""; }
};

} // namespace my

// Template function that requires an Iterable type
template <Iterable T>
typename T::value_type findMaxWithMyTrait(const T &collection) {
  // std::numeric_limits<double>::min(); is a trait
  typename T::value_type largest =
      my::numeric_limits<typename T::value_type>::min();
  for (const auto &ele : collection) {
    if (ele > largest)
      largest = ele;
  }
  return largest;
}

int main() {
  std::array<int, 5> arr = {123, 345, 567, 0, -1};
  std::list<float> lst = {3.14, 1.414, 2.71, -0, 2147483647.0};
  std::deque<std::string> deq = {"Hello", "world", "0xdead", "0xbeef", "!~"};

  cout << "findMax(arr): " << findMax(arr) << "\n";
  cout << "findMax(lst): " << findMax(lst) << "\n";
  cout << "findMax(deq): " << findMax(deq) << "\n";

  cout << "findMaxWithMyTrait(arr): " << findMaxWithMyTrait(arr) << "\n";
  cout << "findMaxWithMyTrait(lst): " << findMaxWithMyTrait(lst) << "\n";
  cout << "findMaxWithMyTrait(deq): " << findMaxWithMyTrait(deq) << "\n";

  return 0;
}