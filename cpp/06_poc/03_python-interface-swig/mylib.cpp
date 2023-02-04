#include <iostream>
#include "mylib.h"

using namespace std;

DepartmentHandler::DepartmentHandler(Department& dept, uint32_t studentCount) :
    _dept(dept) {
    // WTF is _dept(dept)??:
    // https://stackoverflow.com/questions/66393445/error-constructor-must-explicitly-initialize-reference-member
    srand(time(NULL));
    this->studentCount = studentCount;
    students = vector<Student>(studentCount);
}

void DepartmentHandler::prepareStudentData() {
    
    for (size_t i = 0; i < studentCount; ++i) {
        students[i] = Student(string("Test Name"), 1.0 * rand() / rand(),
            1.0 * rand() / rand(), 1.0 * rand() / rand(), 1.0 * rand() / rand());
    }
}

uint32_t DepartmentHandler::GetStudentCount() {
    return studentCount;
}

void DepartmentHandler::start() {
        for (uint32_t i = 0; i < studentCount; ++i) {
            _dept.onStudentIterated(students[i]);
        }
    }
