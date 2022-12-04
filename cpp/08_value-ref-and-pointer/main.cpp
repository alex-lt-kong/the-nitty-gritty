#include <stdio.h>

class SimpleClass {
private:
  int val;
public:
SimpleClass(int val) {
  this->val = val;  
}
~SimpleClass() {}

void setNewVal(int new_val) {
  this->val = new_val;
}

int getVal() {
  return this->val;
}

SimpleClass(const SimpleClass& obj) {
  this->val = obj.val;
  printf("Copy constructor is called()\n");
}
};

void a_naive_function(SimpleClass simple_obj) {
  simple_obj.setNewVal(1234);
  printf("simple_obj.getVal()@a_naive_function(): %d\n", simple_obj.getVal());
}

void a_naive_but_by_ref_function(SimpleClass& simple_obj) {
  simple_obj.setNewVal(1234);
  printf("simple_obj.getVal()@a_naive_but_by_ref_function(): %d\n", simple_obj.getVal());
}

void a_naive_but_by_ref_with_ptr_function(SimpleClass& simple_obj) {
  (&simple_obj)->setNewVal(1234);
  printf("simple_obj.getVal()@a_naive_but_by_ref_with_ptr_function(): %d\n", (&simple_obj)->getVal());
}

void a_naive_but_by_pointer_function(SimpleClass* simple_obj) {
  simple_obj->setNewVal(1234);
  printf("simple_obj.getVal()@a_naive_but_by_pointer_function(): %d\n", simple_obj->getVal());
}

void test_pass_arguments_by_value() {
  SimpleClass simple_obj = SimpleClass(10);
  printf("simple_obj.getVal()@test_pass_arguments_by_value(): %d\n", simple_obj.getVal());
  a_naive_function(simple_obj);
  printf("simple_obj.getVal()@test_pass_arguments_by_value(): %d\n", simple_obj.getVal());
  printf("\n");
}

void test_pass_arguments_by_reference() {
  SimpleClass simple_obj = SimpleClass(10);
  printf("simple_obj.getVal()@test_pass_arguments_by_value(): %d\n", simple_obj.getVal());
  a_naive_but_by_ref_function(simple_obj);
  printf("simple_obj.getVal()@test_pass_arguments_by_value(): %d\n", simple_obj.getVal());
  printf("\n");
}

void test_pass_arguments_by_reference_with_pointer() {
  SimpleClass simple_obj = SimpleClass(10);
  printf("simple_obj.getVal()@test_pass_arguments_by_value(): %d\n", simple_obj.getVal());
  a_naive_but_by_ref_with_ptr_function(simple_obj);
  printf("simple_obj.getVal()@test_pass_arguments_by_value(): %d\n", simple_obj.getVal());
  printf("\n");
}

void test_pass_arguments_by_pointer() {
  SimpleClass simple_obj = SimpleClass(10);
  printf("simple_obj.getVal()@test_pass_arguments_by_value(): %d\n", simple_obj.getVal());
  a_naive_but_by_pointer_function(&simple_obj);
  printf("simple_obj.getVal()@test_pass_arguments_by_value(): %d\n", simple_obj.getVal());
  printf("\n");
}

int main() {
  test_pass_arguments_by_value();
  test_pass_arguments_by_reference();
  test_pass_arguments_by_reference_with_pointer();
  test_pass_arguments_by_pointer();

  int target = 666;
  int &target_ref = target;  // target_ref is initialized to refer to target, i.e., it is a reference to target.
  int* target_ptr = &target; // target_ptr is a pointer, whose value is the address of target.
  printf("%d, %d, %d\n", target, target_ref, *target_ptr);
  ++target;
  printf("%d, %d, %d\n", target, target_ref, *target_ptr);
  return 0;
}

