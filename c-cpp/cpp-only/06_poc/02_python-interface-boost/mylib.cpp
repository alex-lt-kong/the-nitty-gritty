#include <iostream>
#include "mylib.h"

using namespace std;

DepartmentHandler::DepartmentHandler() {}


DepartmentHandler::DepartmentHandler(uint32_t studentCount, boost::python::object object) {
    srand(time(NULL));
    this->object = object;
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
    auto func = object.attr("onStudentIterated");
    for (uint32_t i = 0; i < studentCount; ++i) {
        func(students[i]);
    }
}

void DepartmentHandler::onStudentIterated(Student stu) {
    cout << stu.score1 << endl;
}

DepartmentHandler::~DepartmentHandler() {}

using namespace boost::python;

BOOST_PYTHON_MODULE(mylib)
{
    boost::python::class_<DepartmentHandler>("StudentHandler", init<uint32_t, object>())
        .def("start", &DepartmentHandler::start)
        .def("prepareStudentData", &DepartmentHandler::prepareStudentData)
        .def("onStudentIterated", &DepartmentHandler::onStudentIterated)
    ;
    boost::python::class_<Student>("Student")
        .def_readwrite("name", &Student::name)
        .def_readwrite("score1", &Student::score1)
        .def_readwrite("score2", &Student::score2)
        .def_readwrite("score3", &Student::score3)
        .def_readwrite("score4", &Student::score4);
    ;
};
