#include <iostream>

using namespace std;

template <typename T> __attribute__((noinline)) T my_max(T a, T b) {
  return a > b ? a : b;
}

std::string helper_gen_random_string(const int len) {
  static const char alphanum[] = "0123456789"
                                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyz";
  std::string tmp_s;
  tmp_s.reserve(len);

  for (int i = 0; i < len; ++i) {
    tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
  }

  return tmp_s;
}

template <typename T> class SimpleClass {
private:
  T first;
  T second;

public:
  SimpleClass(T first, T second) {
    this->first = first;
    this->second = second;
  }
  ~SimpleClass() {}

  T getSum() { return first + second; }

  T getProduct() { return first * second; }

  T getFirst() { return first; }

  T getSecond() { return second; }
};

void partially_supported_templates() {
  SimpleClass<int> MyIntClass(1, 2);
  cout << "int first: " << MyIntClass.getFirst() << ", "
       << "int second: " << MyIntClass.getSecond() << ", "
       << "int getSum():" << MyIntClass.getSum()
       << "int getProduct():" << MyIntClass.getProduct() << endl;

  SimpleClass<string> MyStrClass(helper_gen_random_string(5),
                                 helper_gen_random_string(5));
  cout << "str first: " << MyStrClass.getFirst() << ", "
       << "str second: " << MyStrClass.getSecond() << ", "
       << "str getSum():"
       << MyStrClass.getSum()
       //   << "str getProduct():" << MyStrClass.getProduct() won't work, as
       //   string * string is not defined.
       << endl;
}

void t_must_be_deducible() {

  cout << "t_must_be_deducible() fired" << endl;
  int max_int;
  int a_int = rand(), b_int = rand();
  max_int = my_max(a_int, b_int);
  cout << "max_int: " << max_int << endl;

  double max_dbl;
  double a_dbl = (double)rand() / rand();
  double b_dbl = (double)rand() / rand();
  max_dbl = my_max(a_dbl, b_dbl);
  // max_dbl = my_max(a_dbl, b_int); won't compile, C++ doesn't know which type
  // should be used (Well this example may not be too convining as it should be
  // "easy" for the compiler to figure out that people want to use double as the
  // common type)
  max_dbl = my_max<double>(a_dbl, b_int);
  cout << "max_dbl: " << max_dbl << "\n" << endl;
}

int main() {
  srand(time(NULL));
  t_must_be_deducible();
  partially_supported_templates();
  return 0;
}