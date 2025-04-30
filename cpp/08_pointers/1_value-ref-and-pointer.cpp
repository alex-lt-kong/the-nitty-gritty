#include <iostream>
#include <string>

using namespace std;

class SimpleClass {
private:
  int val;

public:
  SimpleClass(int val) { this->val = val; }
  ~SimpleClass() {}

  void setNewVal(int new_val) { this->val = new_val; }

  int getVal() { return this->val; }

  SimpleClass(const SimpleClass &obj) {
    this->val = obj.val;
    cout << "Copy constructor is called()\n";
  }
};

void a_naive_function(SimpleClass simple_obj) {

  cout << "a_naive_function()\n";
  simple_obj.setNewVal(1234);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "/n" << endl;
}

void a_naive_but_by_ref_function(SimpleClass &simple_obj) {

  cout << "a_naive_but_by_ref_function()\n";
  simple_obj.setNewVal(1234);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "\n" << endl;
}

void a_naive_but_by_ref_with_ptr_function(SimpleClass &simple_obj) {

  cout << "a_naive_but_by_ref_with_ptr_function()\n";
  (&simple_obj)->setNewVal(1234);
  cout << "simple_obj.getVal(): " << (&simple_obj)->getVal() << "\n" << endl;
}

void a_naive_but_by_pointer_function(SimpleClass *simple_obj) {

  cout << "a_naive_but_by_pointer_function()\n";
  simple_obj->setNewVal(1234);
  cout << "simple_obj.getVal(): " << simple_obj->getVal() << "\n" << endl;
}

void test_pass_arguments_by_value() {

  cout << "test_pass_arguments_by_value()\n";
  SimpleClass simple_obj = SimpleClass(10);
  cout << "simple_obj.getVal():" << simple_obj.getVal() << "\n";
  a_naive_function(simple_obj);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "\n" << endl;
}

void test_pass_arguments_by_reference() {

  cout << "test_pass_arguments_by_reference()\n";
  SimpleClass simple_obj = SimpleClass(10);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "\n";
  a_naive_but_by_ref_function(simple_obj);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "\n" << endl;
}

void test_pass_arguments_by_reference_with_pointer() {
  cout << "test_pass_arguments_by_reference_with_pointer()\n";
  SimpleClass simple_obj = SimpleClass(10);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "\n";
  a_naive_but_by_ref_with_ptr_function(simple_obj);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "\n" << endl;
}

void test_pass_arguments_by_pointer() {
  cout << "test_pass_arguments_by_pointer()\n";
  SimpleClass simple_obj = SimpleClass(10);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "\n";
  a_naive_but_by_pointer_function(&simple_obj);
  cout << "simple_obj.getVal(): " << simple_obj.getVal() << "\n" << endl;
}

void test_reference_address_of() {
  cout << "test_reference_address_of()\n";
  int target = 666;
  // target_ref is initialized to refer to target,i.e., it is a reference to
  // target.
  int &target_ref = target;
  // target_ptr is a pointer, whose value is the address of target.
  int *target_ptr = &target;
  cout << target << ", " << target_ref << ", " << *target_ptr << "\n";
  ++target;
  cout << target << ", " << target_ref << ", " << *target_ptr << "\n" << endl;
}

void test_cow_in_stl() {
  cout << "test_cow_in_stl()\n";
  string str1("Hello!");
  string str2 = str1;
  cout << str1.c_str() << ", " << str2.c_str() << "\n";
  cout << str1.data() << ", " << str2.data() << "\n" << endl;
}

int main() {
  test_pass_arguments_by_value();
  test_pass_arguments_by_reference();
  test_pass_arguments_by_reference_with_pointer();
  test_pass_arguments_by_pointer();

  test_reference_address_of();
  test_cow_in_stl();
  return 0;
}
