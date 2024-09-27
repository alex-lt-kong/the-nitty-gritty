#include <cstddef>
#include <fmt/core.h>
#include <iterator>

#define ARR_SIZE 20

class IntegersCollection {
public:
  // An iterator is usually declared inside the class it belongs to
  class Iterator {
  public:
    // std::iterator_traits defines five expected traits of an iterator class:
    // https://en.cppreference.com/w/cpp/iterator/iterator_traits
    using iterator_category =
        std::forward_iterator_tag; // one of the six iterator categories
    using difference_type = std::ptrdiff_t;
    using value_type = int;
    using pointer = value_type *;
    using reference = value_type &;

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
  // IntegersCollection() { static_assert(std::forward_iterator<Iterator>); };
  Iterator begin() { return Iterator(&m_data[0]); }
  Iterator end() { return Iterator(&m_data[ARR_SIZE]); }

private:
  int m_data[ARR_SIZE];
};

template <class T> class GenericCollection {
public:
  // An iterator is usually declared inside the class it belongs to
  class Iterator {
  public:
    // std::iterator_traits defines five expected traits of an iterator class:
    // https://en.cppreference.com/w/cpp/iterator/iterator_traits
    using iterator_category =
        std::forward_iterator_tag; // one of the six iterator categories
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type *;
    using reference = value_type &;

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
  // GenericCollection() { static_assert(std::random_access_iterator<Iterator>);
  // };
  Iterator begin() { return Iterator(&m_data[0]); }
  Iterator end() { return Iterator(&m_data[ARR_SIZE]); }

private:
  T m_data[ARR_SIZE];
};
