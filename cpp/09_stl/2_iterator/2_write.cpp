#include <cstddef>
#include <fmt/core.h>
#include <iterator>

class Integers {
public:
  // An iterator is usually declared inside the class it belongs to
  class Iterator {
  public:
    // one of the six iterator categories we have seen above.
    using iterator_category = std::forward_iterator_tag;
    // using difference_type = std::ptrdiff_t;
    // using value_type = int;
    using pointer = int *;
    using reference = int &;

    Iterator(pointer ptr) : m_ptr(ptr) {}

    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }
    Iterator &operator++() {
      m_ptr++;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }
    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.m_ptr == b.m_ptr;
    };
    friend bool operator!=(const Iterator &a, const Iterator &b) {
      return a.m_ptr != b.m_ptr;
    };

  private:
    pointer m_ptr;
  };

  Iterator begin() { return Iterator(&m_data[0]); }
  Iterator end() { return Iterator(&m_data[20]); }

private:
  int m_data[20];
};

int main() {
  Integers arr;
  int count = 1;
  for (auto itr = arr.begin(); itr != arr.end(); ++itr) {
    count *= 3;
    count %= 1024;
    *itr = count;
  }

  for (auto itr = arr.begin(); itr != arr.end(); ++itr) {
    fmt::print("{}, ", *itr);
  }
  fmt::print("\n");
  return 0;
}