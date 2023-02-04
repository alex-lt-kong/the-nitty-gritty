#include <iostream>
#include "mylib.h"

using namespace std;

StudentHandler::StudentHandler(uint32_t studentCount) {
    srand(time(NULL));
    this->studentCount = studentCount;
    students = vector<Student>(studentCount);
}

void StudentHandler::prepareStudentData() {
    
    for (size_t i = 0; i < studentCount; ++i) {
        students[i] = Student(string("Test Name"), 1.0 * rand() / rand(),
            1.0 * rand() / rand(), 1.0 * rand() / rand(), 1.0 * rand() / rand());
    }
}

uint32_t StudentHandler::GetStudentCount() {
    return studentCount;
}

void StudentHandler::start() {
        for (uint32_t i = 0; i < studentCount; ++i) {
            onStudentIterated(students[i]);
        }
    }

StudentHandler::~StudentHandler() {}