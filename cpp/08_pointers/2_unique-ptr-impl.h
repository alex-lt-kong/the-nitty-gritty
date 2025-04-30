#include <functional>
#include <print>

template<typename T>
class my_unique_ptr {
private:
  T *m_ptr;
  std::allocator<T> allocator = std::allocator<T>();
  std::function<void(T *)> m_deleter;

public:
  my_unique_ptr() noexcept
    : m_ptr(nullptr), m_deleter([](const T *p) {
      std::print("delete p;\n");
      delete p;
    }) {
  }

  explicit my_unique_ptr(T *ptr)
    : m_ptr(ptr), m_deleter([](const T *p) {
      std::print("delete p;\n");
      delete p;
    }) {
  }

  explicit my_unique_ptr(T *ptr, decltype(m_deleter) deleter)
    : m_ptr(ptr), m_deleter(deleter) {
  }

  // Delete copy constructor
  explicit my_unique_ptr(const my_unique_ptr &other) = delete;

  // Move constructor
  my_unique_ptr(my_unique_ptr &&other) noexcept
    : m_ptr(other.m_ptr), m_deleter(other.m_deleter) {
    other.m_ptr = nullptr;
  }

  // Delete copy assignment operator
  my_unique_ptr &operator=(const my_unique_ptr &others) = delete;

  // Move assignment operator
  my_unique_ptr &operator=(my_unique_ptr &&other) noexcept {
    if (this != &other) {
      reset();
      m_ptr = other.m_ptr;
      m_deleter = std::move(other.m_deleter);
      other.m_ptr = nullptr;
    }
    return *this;
  }

  T *operator->() const noexcept { return m_ptr; }

  T operator*() const noexcept { return *m_ptr; }

  bool operator==(T *ptr) { return m_ptr == ptr; }

  bool operator!=(T *ptr) { return !(m_ptr == ptr); }

  T *get() { return m_ptr; }

  ~my_unique_ptr() { reset(); }

  void swap(my_unique_ptr &other) noexcept {
    std::swap(m_ptr, other.m_ptr);
    std::swap(m_deleter, other.m_deleter);
  }

  explicit operator bool() const noexcept { return m_ptr != nullptr; }

  void reset(T *ptr = nullptr) noexcept {
    if (m_ptr) {
      m_deleter(m_ptr);
    }
    m_ptr = ptr;
  }
};

template<typename T, typename... Args>
my_unique_ptr<T> my_make_unique(Args &&... args) {
  return my_unique_ptr<T>(new T(std::forward<Args>(args)...));
}
