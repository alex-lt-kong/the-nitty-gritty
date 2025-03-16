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
    cout << "[Interal] move constructor called, rhs._str: [" << rhs.m_buf << "]"
         << endl;
    m_buf = rhs.m_buf; // "Ownership transfer"
    rhs.m_buf = nullptr;
  }

  // Copy assignment operator
  my_string &operator=(const my_string &rhs) {
    bool reallocation = false;
    if (m_capacity < rhs.m_capacity) {
      reset();
      m_capacity = rhs.m_capacity;
      m_buf = allocator.allocate(m_capacity);
      reallocation = true;
    } else {
      // Let's memset() before memcpy() to avoid offset calculation here...
      std::memset(m_buf, 0, m_capacity);
    }
    std::memcpy(m_buf, rhs.m_buf, m_capacity);
    std::println("Copy assignment operator=(const my_string &rhs) called, "
                 "reallocation: {}, m_capacity: {}, m_buf: [{}]",
                 reallocation, m_capacity, m_buf);
    return *this;
  }

  my_string operator+(const my_string &rhs) {
    cout << "[Interal] addition operator called, rhs._str: [" << rhs.m_buf
         << "]" << endl;
    char *temp = (char *)malloc(sizeof(char) *
                                (strlen(rhs.m_buf) + strlen(this->m_buf) + 1));
    if (temp == NULL) {
      throw bad_alloc();
    }
    strcpy(temp, this->m_buf);
    strcpy(temp + strlen(this->m_buf), this->m_buf);

    // memcpy(temp + strlen(this->buf), rhs.buf, strlen(rhs.buf));
    my_string ns(temp);
    free(temp);
    return ns;
  }

  /*
    // Move assignment operator
    my_string operator=(my_string &&rhs) noexcept {
      cout << "[Interal] move assignment operator called, ";
      if (this != &rhs) { // self-assignment protection
        cout << "and it is making an impact, rhs._str: [" << rhs.m_buf << "]"
             << endl;
        free(this->m_buf);
        this->m_buf = rhs.m_buf;
        rhs.m_buf = nullptr;
      } else {
        cout << "but it is doing nothing, rhs._str: [" << rhs.m_buf << "]"
             << endl;
      }
      return *this;
    }
  */
  bool operator==(const my_string &other) const {
    return strcmp(this->m_buf, other.m_buf) == 0;
  }

  bool operator==(const std::string &other) const {
    return strcmp(this->m_buf, other.c_str()) == 0;
  }

  void PrintInternalPointerAddress() {
    cout << "_str: " << (void *)m_buf << endl;
  }

  friend ostream &operator<<(ostream &os, const my_string &ns) {
    os << ns.m_buf;
    return os;
  }

  char *c_str() const noexcept { return m_buf; }

  std::size_t size() const noexcept { return strlen(m_buf); }

  ~my_string() { reset(); }
};

/*
int main() {
  cout << "Test 1\n";
  my_string ns_hel((const char *)"Hello ");
  auto ns_hel2 = ns_hel;
  cout << "ns_hel2: " << ns_hel2 << endl << endl;

  cout << "Test 2\n";
  my_string ns_wor((char *)"world!");
  cout << "ns_hel + ns_wor: " << ns_hel + ns_wor << endl << endl;

  cout << "Test 3\n";
  my_string ns_hw0 = ns_hel + ns_wor;
  // This mostly triggers copy constructor with copy elision
  cout << "ns_hw0: " << ns_hw0 << endl << endl;

  cout << "Test 4\n";
  my_string ns_hw1 = my_string((char *)"Hello world");
  cout << "ns_hw1: " << ns_hw1 << "\n";
  ns_hw1 = ns_hw1;
  cout << "ns_hw1: " << ns_hw1 << "\n";
  ns_hw1 = my_string((char *)"Goodbye");
  cout << "ns_hw1: " << ns_hw1 << endl << endl;

  cout << "Test 5\n";
  my_string ns_fb = my_string((char *)"foobar");
  cout << "ns_fb: " << ns_fb << "\n";
  ns_fb = ns_hw1;
  cout << "ns_fb: " << ns_fb << endl << endl;

  cout << "Test 6\n";
  my_string ns_fh = my_string((char *)"foobar") + my_string((char *)"Hello!");
  cout << "ns_fh: " << ns_fh << endl << endl;

  cout << "Test 7\n";
  ns_fb.PrintInternalPointerAddress();
  my_string ns_fb1 = std::move(ns_fb);
  ns_fb.PrintInternalPointerAddress();
  ns_fb1.PrintInternalPointerAddress();
  return 0;
}
*/