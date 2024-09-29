#include <cstddef>
#include <fmt/core.h>
#include <iterator>

template <size_t N> class IntegersCollection {
public:
  IntegersCollection() {
    // statically check whether intCollection has a proper forward_iterator
    // https://en.cppreference.com/w/cpp/iterator/forward_iterator
    static_assert(std::forward_iterator<IntegersCollection<1>::iterator>);
  }
  // An iterator is usually declared inside the class it belongs to
  class iterator {
  public:
    // std::iterator_traits defines five expected traits of an iterator class:
    // https://en.cppreference.com/w/cpp/iterator/iterator_traits
    using iterator_category =
        std::forward_iterator_tag; // one of the six iterator categories
                                   //  using difference_type = std::ptrdiff_t;
    using value_type = int;
    using pointer = value_type *;
    using reference = value_type &;
    using difference_type = std::ptrdiff_t;

    // Default constructor is required to pass forward_iterator assertion
    iterator() { throw std::runtime_error("Not implemented"); }
    iterator(pointer ptr) : m_ptr(ptr) {}

    reference operator*() const { return *m_ptr; }

    // prefix increment operator, i.e, ++obj;
    iterator &operator++() {
      m_ptr++;
      return *this;
    }

    // postfix increment operator, i.e., obj++;
    // (int) is a dummy parameter:
    // https://stackoverflow.com/questions/12740378/why-use-int-as-an-argument-for-post-increment-operator-overload
    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const iterator &a, const iterator &b) {
      return a.m_ptr == b.m_ptr;
    };

  private:
    pointer m_ptr;
  };
  // IntegersCollection() { static_assert(std::forward_iterator<Iterator>); };
  iterator begin() { return iterator(&m_data[0]); }
  iterator end() { return iterator(&m_data[N]); }

private:
  int m_data[N];
};

template <typename T, size_t N> class GenericCollection {
public:
  // An iterator is usually declared inside the class it belongs to
  class iterator {
  public:
    // std::iterator_traits defines five expected traits of an iterator class:
    // https://en.cppreference.com/w/cpp/iterator/iterator_traits
    using iterator_category =
        std::forward_iterator_tag; // one of the six iterator categories
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type *;
    using reference = value_type &;

    // Default constructor is required to pass forward_iterator assertion
    iterator() { throw std::runtime_error("Not implemented"); }
    iterator(pointer ptr) : m_ptr(ptr) {}

    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }
    iterator &operator++() {
      m_ptr++;
      return *this;
    }
    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }
    friend bool operator==(const iterator &a, const iterator &b) {
      return a.m_ptr == b.m_ptr;
    };
    friend bool operator!=(const iterator &a, const iterator &b) {
      return a.m_ptr != b.m_ptr;
    };

  private:
    pointer m_ptr;
  };
  // GenericCollection() { static_assert(std::random_access_iterator<Iterator>);
  // };
  iterator begin() { return iterator(&m_data[0]); }
  iterator end() { return iterator(&m_data[N]); }

private:
  T m_data[N];
};
