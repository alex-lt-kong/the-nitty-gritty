#include <iostream>
#include <string.h>
using namespace std;

class NaiveString {
private:
  char *buf;

public:
  // Must be a null-terminated C-string
  NaiveString(const char *str) {
    cout << "[Interal] constructor called" << endl;
    buf = (char *)malloc(sizeof(char) * (strlen(str) + 1));
    if (buf == NULL) {
      throw bad_alloc();
    }
    // memcpy(buf, str, strlen(str)); won't work as buf is not initialized to 0
    // in malloc()
    strcpy(buf, str);
  }

  // Copy constructor
  NaiveString(const NaiveString &rhs) {
    cout << "[Interal] copy constructor called, rhs._str: [" << rhs.buf << "]"
         << endl;
    buf = (char *)malloc(sizeof(char) * (strlen(rhs.buf) + 1));
    if (buf == NULL) {
      throw bad_alloc();
    }
    // memcpy(buf, str, strlen(str)); won't work as buf is not initialized to 0
    // in malloc()
    strcpy(buf, rhs.buf);
  }

  // Move constructor
  NaiveString(NaiveString &&rhs) noexcept {
    cout << "[Interal] move constructor called, rhs._str: [" << rhs.buf << "]"
         << endl;
    buf = rhs.buf; // "Ownership transfer"
    rhs.buf = nullptr;
  }

  NaiveString operator+(const NaiveString &rhs) {
    cout << "[Interal] addition operator called, rhs._str: [" << rhs.buf << "]"
         << endl;
    char *temp = (char *)malloc(sizeof(char) *
                                (strlen(rhs.buf) + strlen(this->buf) + 1));
    if (temp == NULL) {
      throw bad_alloc();
    }
    strcpy(temp, this->buf);
    strcpy(temp + strlen(this->buf), this->buf);

    // memcpy(temp + strlen(this->buf), rhs.buf, strlen(rhs.buf));
    NaiveString ns(temp);
    free(temp);
    return ns;
  }

  // Copy assignment operator
  NaiveString operator=(const NaiveString &rhs) {

    cout << "[Interal] copy assignment operator called, rhs._str: [" << rhs.buf
         << "]" << endl;
    char *temp = (char *)malloc(sizeof(char) * (strlen(rhs.buf) + 1));
    if (temp == NULL) {
      throw bad_alloc();
    }
    strcpy(temp, this->buf);
    free(this->buf);
    this->buf = temp;
    return *this;
  }

  // Move assignment operator
  NaiveString operator=(NaiveString &&rhs) noexcept {

    cout << "[Interal] move assignment operator called, ";
    if (this != &rhs) { // self-assignment protection
      cout << "and it is making an impact, rhs._str: [" << rhs.buf << "]"
           << endl;
      free(this->buf);
      this->buf = rhs.buf;
      rhs.buf = nullptr;
    } else {
      cout << "but it is doing nothing, rhs._str: [" << rhs.buf << "]" << endl;
    }
    return *this;
  }

  void PrintInternalPointerAddress() {
    cout << "_str: " << (void *)buf << endl;
  }

  friend ostream &operator<<(ostream &os, const NaiveString &ns) {
    os << ns.buf;
    return os;
  }

  ~NaiveString() { free(this->buf); }
};

int main() {
  cout << "Test 1\n";
  NaiveString ns_hel((const char *)"Hello ");
  auto ns_hel2 = ns_hel;
  cout << "ns_hel2: " << ns_hel2 << endl << endl;

  cout << "Test 2\n";
  NaiveString ns_wor((char *)"world!");
  cout << "ns_hel + ns_wor: " << ns_hel + ns_wor << endl << endl;

  cout << "Test 3\n";
  NaiveString ns_hw0 = ns_hel + ns_wor;
  // This mostly triggers copy constructor with copy elision
  cout << "ns_hw0: " << ns_hw0 << endl << endl;

  cout << "Test 4\n";
  NaiveString ns_hw1 = NaiveString((char *)"Hello world");
  cout << "ns_hw1: " << ns_hw1 << "\n";
  ns_hw1 = ns_hw1;
  cout << "ns_hw1: " << ns_hw1 << "\n";
  ns_hw1 = NaiveString((char *)"Goodbye");
  cout << "ns_hw1: " << ns_hw1 << endl << endl;

  cout << "Test 5\n";
  NaiveString ns_fb = NaiveString((char *)"foobar");
  cout << "ns_fb: " << ns_fb << "\n";
  ns_fb = ns_hw1;
  cout << "ns_fb: " << ns_fb << endl << endl;

  cout << "Test 6\n";
  NaiveString ns_fh =
      NaiveString((char *)"foobar") + NaiveString((char *)"Hello!");
  cout << "ns_fh: " << ns_fh << endl << endl;

  cout << "Test 7\n";
  ns_fb.PrintInternalPointerAddress();
  NaiveString ns_fb1 = std::move(ns_fb);
  ns_fb.PrintInternalPointerAddress();
  ns_fb1.PrintInternalPointerAddress();
  return 0;
}