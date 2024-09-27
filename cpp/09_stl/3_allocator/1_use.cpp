#include <iostream>
#include <memory>

int main() {
  auto alloc1 = std::allocator<int>();
  static_assert(std::is_same_v<int, decltype(alloc1)::value_type>);
  size_t sz = 1;
  auto p1 = alloc1.allocate(sz);
  *p1 = 3;
  std::cout << "p1: " << p1 << ", *p1: " << *p1 << "\n";
  alloc1.deallocate(p1, sz);

  sz = 10;
  auto arr = alloc1.allocate(sz);
  for (size_t i = 0; i < sz; ++i) {
    arr[i] = i;
  }
  for (size_t i = 0; i < sz; ++i) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
  alloc1.deallocate(arr, sz);
  return 0;
}