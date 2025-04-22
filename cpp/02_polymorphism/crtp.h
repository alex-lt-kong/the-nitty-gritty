#ifndef CRTP_H
#define CRTP_H

#include <print>

template <typename T> class Base {
public:
  void show() { static_cast<T *>(this)->showImpl(); }
  void showImpl() { std::println("Base::show() called."); }
};

class EssentiallyBase : public Base<EssentiallyBase> {};

class Derived : public Base<Derived> {
public:
  void showImpl() { std::println("Derived::show() called."); }
};

template <typename T> void show(Base<T> &obj) { obj.show(); }

#endif // CRTP_H
