#include <memory>
#include <print>   // C++23 printing header

struct Resource {
    int value;
};

int main() {
    // Create a shared resource with an initial owning pointer.
    auto res = std::make_shared<Resource>();
    res->value = 100;
    std::println("Initial use_count: {}", res.use_count());  // Expected: 1

    // Create additional shared pointers pointing to the same resource.
    auto shared1 = res;
    std::println("After copying to shared1, use_count: {}", res.use_count());  // Expected: 2

    auto shared2 = res;
    std::println("After copying to shared2, use_count: {}", res.use_count());  // Expected: 3

    // Create a scoped shared pointer.
    {
        auto shared3 = res;
        std::println("Inside block, after copying to shared3, use_count: {}", res.use_count());  // Expected: 4
        // shared3 goes out of scope here.
    }
    std::println("After block, use_count: {}", res.use_count());  // Expected: 3

    // Use the resource via one of the shared pointers.
    std::println("Resource value from shared2: {}", shared2->value);

    // Reset one of the shared pointers.
    shared1.reset();
    std::println("After resetting shared1, use_count: {}", res.use_count());  // Expected: 2

    return 0;
}
