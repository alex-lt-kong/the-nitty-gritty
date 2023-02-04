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

class LIBRARY_API StudentHandler {
public:
    uint32_t studentCount;
    vector<Student> students;
    StudentHandler(uint32_t studentCount);
    void prepareStudentData();
    virtual void onStudentIterated(Student stu) = 0;
    uint32_t GetStudentCount();
    virtual ~StudentHandler();
    void start();

};
