#ifndef BIDIRECTIONAL_ITERATOR_IMPL_H
#define BIDIRECTIONAL_ITERATOR_IMPL_H

#include <cstddef>
#include <iterator>
#include <vector>


template<typename T>
class BidirectionalIterator {
public:
    // std::iterator_traits defines five expected traits of an
    // BidirectionalIterator class:
    // https://en.cppreference.com/w/cpp/BidirectionalIterator/iterator_traits
    // `std::forward_iterator_tag` is one of the six BidirectionalIterator
    // categories
    using iterator_category = std::bidirectional_iterator_tag;
    // is used for pointer arithmetic and array indexing, if negative values
    // are possible. Programs that use other types, such as int, may fail
    // on, e.g. 64-bit systems when the index exceeds INT_MAX or if it
    // relies on 32-bit modular arithmetic.
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T *;
    using reference = T &;

    // Default constructor is required to pass forward_iterator assertion
    BidirectionalIterator() : m_ptr(nullptr) {}
    explicit BidirectionalIterator(const pointer ptr) : m_ptr(ptr) {}

    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }
    BidirectionalIterator &operator++() {
        ++m_ptr;
        return *this;
    }
    BidirectionalIterator operator++(int) {
        BidirectionalIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    // Compared with forward_iterator, we only need the following two methods
    BidirectionalIterator &operator--() {
        --m_ptr;
        return *this;
    }
    BidirectionalIterator operator--(int) {
        BidirectionalIterator tmp = *this;
        --(*this);
        return tmp;
    }

    friend bool operator==(const BidirectionalIterator &a,
                           const BidirectionalIterator &b) {
        return a.m_ptr == b.m_ptr;
    }
    friend bool operator!=(const BidirectionalIterator &a,
                           const BidirectionalIterator &b) {
        return a.m_ptr != b.m_ptr;
    }

private:
    pointer m_ptr;
};


template<typename T>
class MyContainerWithBidirectionalIterator {
public:
    using iterator = BidirectionalIterator<T>;
    explicit MyContainerWithBidirectionalIterator(const std::vector<T> &data) :
        m_data(data) {}
    iterator begin() { return iterator(m_data.data()); }
    iterator end() { return iterator(m_data.data() + m_data.size()); }

private:
    std::vector<T> m_data;
};
#endif // BIDIRECTIONAL_ITERATOR_IMPL_H
