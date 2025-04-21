#ifndef FUNCTION_OVERRIDING_WITH_CRTP_H
#define FUNCTION_OVERRIDING_WITH_CRTP_H
#include <print>

#include <print>

template <typename Derived> class CrtpBase {
public:
  void show() { static_cast<Derived *>(this)->showImpl(); }
  void showImpl() { std::println("CrtpBase::show() called."); }
};

class CrtpDummyDerived : public CrtpBase<CrtpDummyDerived> {};

class CrtpDerived : public CrtpBase<CrtpDerived> {
public:
  void showImpl() { std::println("CrtpDerived::show() called."); }
};

template <typename T> void show(CrtpBase<T> &obj) { obj.show(); }

#endif // FUNCTION_OVERRIDING_WITH_CRTP_H
