#include <iostream>

using namespace std;

class Person {

protected:
    int age_ = -1;

public:
    Person() {}
    Person(int age) {
        this->age_ = age;
    }

    int age() {
        return this->age_;
    }
};


class Adult : public Person {

private:
    const static int minAge_ = 18;
    // if we want to use minAge_ in Person::Person(Adult::minAge_ + 33),
    // minAge_ canNOT be a member of instance but have to be a member of the class--C++ standards require
    // base class to be initlized first so there isn't a simple way to use this->minAge_ in Person::Person(this->minAge_ + 33)
public:
    Adult():
    Person(Adult::minAge_ + 33) // Call the superclass constructor in the subclass' initialization list. That is we:
    // call Person::Person(66); and then Adult:Adult();
    // Note that is it no as straightfoward to do it the other way around..
    {
        cout << "before assignment:" << this->age_ << endl;
        this->age_ = 23;
        cout << "after assignment:" << this->age_ << endl;
    }
};


int main() {
    Person nobody = Person(19);
    cout << nobody.age() << endl;
    Adult oldNobody = Adult();
    cout << oldNobody.age() << endl;
    return 0;
}