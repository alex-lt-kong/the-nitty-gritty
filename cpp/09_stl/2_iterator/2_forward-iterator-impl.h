#ifndef FORWARD_ITERATOR_IMPL_H
#define FORWARD_ITERATOR_IMPL_H

#include <cstddef>
#include <iterator>
#include <vector>

template<typename T>
class ForwardIterator {
public:
    // std::iterator_traits defines five expected traits of an iterator
    // class: https://en.cppreference.com/w/cpp/iterator/iterator_traits
    // `std::forward_iterator_tag` is one of the six iterator categories
    using iterator_category = std::forward_iterator_tag;
    // is used for pointer arithmetic and array indexing, if negative values
    // are possible. Programs that use other types, such as int, may fail
    // on, e.g. 64-bit systems when the index exceeds INT_MAX or if it
    // relies on 32-bit modular arithmetic.
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T *;
    using reference = T &;

    // Default constructor is required to pass forward_iterator assertion
    ForwardIterator() : m_ptr(nullptr) {}
    explicit ForwardIterator(const pointer ptr) : m_ptr(ptr) {}

    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }
    ForwardIterator &operator++() {
        ++m_ptr;
        return *this;
    }
    ForwardIterator operator++(int) {
        ForwardIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    // friend declaration grants a function or another class access to
    // private and protected members of the class where the friend
    // declaration appears.
    // A friend function is not considered a member function, even if we can
    // define it within a class.
    // https://stackoverflow.com/a/49131233/19634193
    friend bool operator==(const ForwardIterator &a, const ForwardIterator &b) {
        return a.m_ptr == b.m_ptr;
    }
    /* Functionally, the below version also works, but it will fail
    static_assert(std::forward_iterator<GenericCollection<T>::iterator>);
    bool operator==(const iterator &rhs) {
        return this->m_ptr == rhs.m_ptr;
    }
    */
    friend bool operator!=(const ForwardIterator &a, const ForwardIterator &b) {
        return a.m_ptr != b.m_ptr;
    }

private:
    pointer m_ptr;
};

template<typename T>
class MyContainerWithForwardIterator {
public: // An iterator is usually declared inside the class it belongs to
    using iterator = ForwardIterator<T>;
    explicit MyContainerWithForwardIterator(const std::vector<T> &data) :
        m_data(data) {}
    iterator begin() { return iterator(m_data.data()); }
    iterator end() { return iterator(m_data.data() + m_data.size()); }

private:
    std::vector<T> m_data;
};
#endif // FORWARD_ITERATOR_IMPL_H
