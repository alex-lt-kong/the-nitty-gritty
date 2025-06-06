#ifndef FUNCTION_OVERRIDING_H
#define FUNCTION_OVERRIDING_H

#include <print>

class Base {
public:
  virtual ~Base() = default;
  virtual void show() { std::println("Base::show() called."); }
  /* no virtual here */ void showWithoutVirtual() {
    std::println("Base::showWithoutVirtual() called.");
  }
};

class Derived : public Base {
public:
  void show() override { std::println("Derived::show() called."); }
  void showWithoutVirtual() /* can't override */ {
    std::println("Derived::showWithoutVirtual() called.");
  }
};

// show() has no way to ascertain which show() to call at compile time
inline void show(Base &obj) { obj.show(); }
inline void showWithoutVirtual(Base &obj) { obj.showWithoutVirtual(); }

#endif // FUNCTION_OVERRIDING_H
