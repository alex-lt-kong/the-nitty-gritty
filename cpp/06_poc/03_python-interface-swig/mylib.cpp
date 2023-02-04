#include <iostream>
#include "mylib.h"

using namespace std;

StudentHandler::StudentHandler() {
    studentCount = 0;
    stu = Student();
}

uint32_t StudentHandler::GetStudentCount() {
    return studentCount;
}

StudentHandler::~StudentHandler() {}