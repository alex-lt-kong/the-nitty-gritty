// In C++, perfect forwarding is the act of passing a function’s parameters to
// another function while preserving its reference category (e.g.,
// rvalue/lvalue/etc). It is commonly used by wrapper methods that want to pass
// their parameters through to another function, often a constructor.
// https://devblogs.microsoft.com/oldnewthing/20230727-00/?p=108494

#include <iostream>
#include <string>

template <typename T> void g(T &&t) {
  std::cout << "Rvalue overload called, msg: " << t << "\n";
}

template <typename T> void g(T &t) {
  std::cout << "Lvalue overload called, msg: " << t << "\n";
}

template <typename T> void non_forwarding_func(T &&t) { g(t); }

// perfectly forwarding func always takes an T &&t
template <typename T> void perfectly_forwarding_func(T &&t) {
  g(std::forward<T>(t));
}

// A special class prepared to demo "forwarding reference" (a.k.a. "universal
// reference")
// https://stackoverflow.com/questions/39552272/is-there-a-difference-between-universal-references-and-forwarding-references
template <typename T> class DummyClass {
public:
  static void perfectly_forwarding_func_T(T &&t) { g(std::forward<T>(t)); }

  template <typename U>
  // We dont really need both requirements, this is just for demo purpose
    requires std::assignable_from<T &, U> || std::convertible_to<U, T>
  static void perfectly_forwarding_func_U(U &&t) {
    g(std::forward<U>(t));
  }
};

int main() {
  std::string text = "Hello, world! from variable text";
  std::cout << "WithOUT perfect forwarding: everything becomes Lvalue:\n";
  non_forwarding_func(text);
  non_forwarding_func(std::string("Hello, world! from temporary string"));
  non_forwarding_func(std::move(text));
  std::cout << "\n";

  std::cout << "WITH perfect forwarding: an Lvalue is still an Lvalue, an "
               "Rvalue is still an Rvalue:\n";
  perfectly_forwarding_func(text);
  perfectly_forwarding_func(std::string("Hello, world! from temporary string"));
  perfectly_forwarding_func(std::move(text));
  std::cout << "\n";

  std::cout << "Funny case: if you want perfect forwarding to work, you must "
               "leave the type T/U auto deduced";
  // This is used in lockfree SPSC Queue:
  // https://github.com/alex-lt-kong/lockfree-toolkit/blob/05f1cad4023f0c2f87335beb237e8d3efb768a04/ringbuffer/ringbuffer-spsc-impl.h#L49-L59
  // Special case: these three statements won't work! Universal reference only works when its type is auto deduced.
  // - Stmt1: DummyClass already deduced T, so for perfectly_forwarding_func_T() T is not auto deduced
  // DummyClass<std::string>::perfectly_forwarding_func_T(text);
  // - Stmt2: DummyClass already deduced T, and we specified U, so for perfectly_forwarding_func_U() U is not auto deduced
  // DummyClass<std::string>::perfectly_forwarding_func_U<std::string>(text);
  // - Stmt3: perfectly_forwarding_func's T is also specified, so it is not auto deduced
  // perfectly_forwarding_func<std::string>(text);
  DummyClass<std::string>::perfectly_forwarding_func_U(text);
  DummyClass<std::string>::perfectly_forwarding_func_U(std::move(text));

  std::cout << "\n";
  return 0;
}

/*
main() prints:

WithOUT perfect forwarding: everything becomes Lvalue:
Lvalue overload called, msg: Hello, world! from variable text
Lvalue overload called, msg: Hello, world! from temporary string
Lvalue overload called, msg: Hello, world! from variable text

WITH perfect forwarding: an Lvalue is still an Lvalue, an Rvalue is still an
Rvalue: Lvalue overload called, msg: Hello, world! from variable text Rvalue
overload called, msg: Hello, world! from temporary string Rvalue overload
called, msg: Hello, world! from variable text

Funny case: if you want perfect forwarding to work, you must leave the type T/U
auto deducedLvalue overload called, msg: Hello, world! from variable text Rvalue
overload called, msg: Hello, world! from variable text

 */