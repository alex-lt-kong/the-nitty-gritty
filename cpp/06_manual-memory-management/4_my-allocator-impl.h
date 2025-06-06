#ifndef MANUAL_MEMORY_MANAGEMENT_4_MY_ALLOCATOR_H
#define MANUAL_MEMORY_MANAGEMENT_4_MY_ALLOCATOR_H

#include <memory>
#include <iostream>

namespace My {
    template<typename T>
    struct Allocator {
        using value_type = T;

        Allocator() = default;

        static T *allocate(const std::size_t n) {
            std::cout << "Allocating " << n << " objects\n";
            return static_cast<T *>(::operator new(n * sizeof(T)));
        }

        static void deallocate(T *p, const std::size_t n) {
            std::cout << "Deallocating " << n << " objects\n";
            ::operator delete(p);
        }
    };
}

#endif //MANUAL_MEMORY_MANAGEMENT_4_MY_ALLOCATOR_H
