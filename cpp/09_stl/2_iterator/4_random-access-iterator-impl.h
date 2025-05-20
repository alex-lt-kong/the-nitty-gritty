#ifndef RANDOM_ACCESS_ITERATOR_IMPL_H
#define RANDOM_ACCESS_ITERATOR_IMPL_H

#include <cstddef>
#include <iterator>
#include <vector>


template<typename T>
class RandomAccessIterator {
    using RAI = RandomAccessIterator;

public:
    // std::iterator_traits defines five expected traits of an
    // RandomAccessIterator class:
    // https://en.cppreference.com/w/cpp/iterator/iterator_traits
    // `std::forward_iterator_tag` is one of the six RandomAccessIterator
    // categories
    using iterator_category = std::random_access_iterator_tag;
    // is used for pointer arithmetic and array indexing, if negative values
    // are possible. Programs that use other types, such as int, may fail
    // on, e.g. 64-bit systems when the index exceeds INT_MAX or if it
    // relies on 32-bit modular arithmetic.
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T *;
    using reference = T &;

    // Default constructor is required to pass forward_iterator assertion
    RandomAccessIterator() : m_ptr(nullptr) {}
    explicit RandomAccessIterator(const pointer ptr) : m_ptr(ptr) {}

    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }
    RAI &operator++() {
        ++m_ptr;
        return *this;
    }
    RAI operator++(int) {
        RAI tmp = *this;
        ++(*this);
        return tmp;
    }
    RAI &operator--() {
        --m_ptr;
        return *this;
    }
    RAI operator--(int) {
        RAI tmp = *this;
        --(*this);
        return tmp;
    }


    RAI &operator+=(difference_type n) {
        m_ptr += n;
        return *this;
    }
    RAI &operator-=(difference_type n) {
        m_ptr -= n;
        return *this;
    }

    reference operator[](difference_type n) const { return *(m_ptr + n); }

    // All these friend methods are required by
    // static_assert(std::random_access_iterator<decltype(mc)::iterator>);
    // the requirements are defined in iterator_concept.h
    friend difference_type operator-(const RAI &lhs, const RAI &rhs) {
        return lhs.m_ptr - rhs.m_ptr;
    }
    friend RAI operator-(const RAI &it, difference_type n) {
        return RAI(it.m_ptr - n);
    }
    friend auto operator<=>(const RAI &lhs, const RAI &rhs) = default;
    friend RAI operator+(const RAI &it, difference_type n) {
        return RAI(it.m_ptr + n);
    }

    friend RAI operator+(difference_type n, const RAI &it) {
        return it + n; // Simply delegate to the existing operator+
    }


private:
    pointer m_ptr;
};


template<typename T>
class MyContainerWithRandomAccessIterator {
public:
    using iterator = RandomAccessIterator<T>;
    explicit MyContainerWithRandomAccessIterator(const std::vector<T> &data) :
        m_data(data) {}
    iterator begin() { return iterator(m_data.data()); }
    iterator end() { return iterator(m_data.data() + m_data.size()); }

private:
    std::vector<T> m_data;
};
#endif // RANDOM_ACCESS_ITERATOR_IMPL_H
