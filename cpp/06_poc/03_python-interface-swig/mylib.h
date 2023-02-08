#include <stdint.h>
#include <vector>

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
    inline Student(string name, double score1, double score2,
      double score3, double score4) {
        this->name = name;
        this->score1 = score1;
        this->score2 = score2;
        this->score3 = score3;
        this->score4 = score4;
    }
};

class LIBRARY_API Department {
public:
    // void onStudentIterated(Student stu) {cout<<"default"<<endl;};
    virtual void onStudentIterated(Student& stu) = 0;
    virtual ~Department() {};
};


class LIBRARY_API DepartmentHandler {
  
public:
    Department& _dept;
    uint32_t studentCount;
    vector<Student> students;
    DepartmentHandler(Department& dept, uint32_t studentCount);
    void prepareStudentData();    
    uint32_t GetStudentCount();
    void start();

};
