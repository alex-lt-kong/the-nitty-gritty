#include <iostream>
#include "mylibs.h"

using namespace std;

MyClass::MyClass() {
    this->Id = 31415;
    this->Name = "MyObjectName";
    this->PhoneNumber = 1234567890;
    this->Scores = {78.9, -78.9, 12.3, 3.14, 1.414};
}

void MyClass::Print() {
    cout << "Id: " << Id
         << "\nName: " << Name
         << "\nPhoneNumber: " << PhoneNumber
         << endl;
}

MyClass::~MyClass() {}
