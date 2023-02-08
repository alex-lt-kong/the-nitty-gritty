#include <stdint.h>
#include <vector>
#include <boost/python.hpp>

using namespace std;

#if defined(_WIN32)
  #define LIBRARY_API __declspec(dllexport)
#else
  #define LIBRARY_API
#endif

class LIBRARY_API Student {
public:
    string name;
    double score1;
    double score2;
    double score3;
    double score4;
    inline Student() {};
    inline Student(string name, double score1, double score2, double score3, double score4) {
        this->name = name;
        this->score1 = score1;
        this->score2 = score2;
        this->score3 = score3;
        this->score4 = score4;
    }
};

class LIBRARY_API DepartmentHandler {
public:
    boost::python::object object;
    uint32_t studentCount;
    uint32_t iterCount;
    vector<Student> students;
    DepartmentHandler(uint32_t studentCount, boost::python::object object);
    DepartmentHandler();
    void prepareStudentData();
    void onStudentIterated(Student stu);
    uint32_t GetStudentCount();
    virtual ~DepartmentHandler();
    void start();

};
