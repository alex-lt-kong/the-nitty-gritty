#include <iostream>

using namespace std;

class Person {

protected:
  int m_age = -1;

public:
  Person() {}
  Person(int age) { this->m_age = age; }

  int age() { return this->m_age; }
};

class Adult : public Person {

private:
  const static int m_min_age = 18;
  // if we want to use m_min_age in Person::Person(Adult::m_min_age + 33),
  // m_min_age canNOT be a member of instance but have to be a member of the
  // class--C++ standards require base class to be initlized first so there
  // isn't a simple way to use this->m_min_age in Person::Person(this->m_min_age
  // + 33)
public:
  Adult()
      : Person(Adult::m_min_age +
               33) // Call the superclass constructor in the subclass'
                   // initialization list. That is we:
  // call Person::Person(66); and then Adult:Adult();
  // Note that is it no as straightfoward to do it the other way around..
  {
    cout << "before assignment:" << this->m_age << endl;
    this->m_age = 23;
    cout << "after assignment:" << this->m_age << endl;
  }
};

int main() {
  Person nobody = Person(19);
  cout << nobody.age() << endl;
  Adult oldNobody = Adult();
  cout << oldNobody.age() << endl;
  return 0;
}