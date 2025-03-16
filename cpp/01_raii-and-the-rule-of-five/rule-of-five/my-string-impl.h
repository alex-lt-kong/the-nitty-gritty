#include <cstring>
#include <iostream>
#include <print>

using namespace std;

class my_string {
private:
  std::allocator<char> allocator = std::allocator<char>();
  std::size_t m_capacity;
  char *m_buf = nullptr;

  void reset() {
    allocator.deallocate(m_buf, m_capacity);
    m_capacity = 0;
  }

public:
  my_string() : m_capacity(1), m_buf(allocator.allocate(m_capacity)) {
    m_buf[0] = '\0';
    std::println("Ctor my_string() called, "
                 "m_capacity: {}, m_buf: [{}]",
                 m_capacity, m_buf);
  }

  // Must be a null-terminated C-string
  my_string(const char *str)
      : m_capacity(std::bit_ceil(strlen(str) + 1)),
        m_buf(allocator.allocate(m_capacity)) {
    strcpy(m_buf, str);
    std::println("Ctor my_string(const char *str) called, "
                 "m_capacity: {}, m_buf: [{}]",
                 m_capacity, m_buf);
  }

  // Copy constructor
  my_string(const my_string &rhs)
      : m_capacity(rhs.m_capacity), m_buf(allocator.allocate(m_capacity)) {
    std::memcpy(m_buf, rhs.m_buf, m_capacity);
    std::println("Copy ctor my_string(const my_string &rhs) called, "
                 "m_capacity: {}, m_buf: [{}]",
                 m_capacity, m_buf);
  }

  // Move constructor
  my_string(my_string &&rhs) noexcept {
    m_capacity = rhs.m_capacity;
    m_buf = rhs.m_buf;
    rhs.m_buf = nullptr;
    rhs.m_capacity = 0;
    std::println("Move ctor my_string(my_string &&rhs) called, "
                 "m_capacity: {}, m_buf: [{}]",
                 m_capacity, m_buf);
  }

  // Move constructor
  my_string(char *&&rhs, std::size_t capacity) noexcept
      : m_capacity(capacity), m_buf(rhs) {
    rhs = nullptr;
    std::println("Move ctor my_string(char *&&rhs) called, "
                 "m_capacity: {}, m_buf: [{}]",
                 m_capacity, m_buf);
  }

  // Copy assignment operator
  my_string &operator=(const my_string &rhs) {
    bool reallocation = false;
    if (this != &rhs) {
      if (m_capacity < rhs.m_capacity) {
        reset();
        m_capacity = rhs.m_capacity;
        m_buf = allocator.allocate(m_capacity);
        reallocation = true;
      } else {
        // Let's memset() before memcpy() to avoid offset calculation here...
        std::memset(m_buf, 0, m_capacity);
      }
      std::memcpy(m_buf, rhs.m_buf, rhs.m_capacity);
    }
    std::println("Copy assignment operator=(const my_string &rhs) called, "
                 "reallocation: {}, m_capacity: {}, m_buf: [{}]",
                 reallocation, m_capacity, m_buf);
    return *this;
  }

  // Move assignment operator
  my_string &operator=(my_string &&rhs) noexcept {
    if (this != &rhs) { // self-assignment protection
      reset();
      m_capacity = rhs.m_capacity;
      m_buf = rhs.m_buf;
      rhs.m_capacity = 0;
      rhs.m_buf = nullptr;
    }
    std::println(
        "Move assignment my_string &operator=(my_string &&rhs) called, "
        "m_capacity: {}, m_buf: [{}]",
        m_capacity, m_buf);
    return *this;
  }

  // Addition operator overloading typically returns a value
  my_string operator+(const my_string &rhs) {
    auto capacity = std::bit_ceil(strlen(m_buf) + strlen(rhs.m_buf) + 1);
    auto buf = allocator.allocate(capacity);

    strcpy(buf, this->m_buf);
    strcat(buf, rhs.m_buf);

    my_string ms{std::move(buf), capacity};
    std::println(
        "Addition operator my_string &operator=(my_string &&rhs) called, "
        "ms.m_capacity: {}, ms.m_buf: [{}]",
        ms.m_capacity, ms.m_buf);
    return ms;
  }

  bool operator==(const my_string &other) const {
    return strcmp(this->m_buf, other.m_buf) == 0;
  }

  bool operator==(const std::string &other) const {
    return strcmp(this->m_buf, other.c_str()) == 0;
  }

  char *c_str() const noexcept { return m_buf; }

  std::size_t size() const noexcept {
    return m_capacity == 0 ? 0 : strlen(m_buf);
  }

  ~my_string() { reset(); }
};