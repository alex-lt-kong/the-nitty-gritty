#include <stdint.h>

using namespace std;

#if defined(_WIN32)
  #define LIBRARY_API __declspec(dllexport)
#else
  #define LIBRARY_API
#endif

class LIBRARY_API Student {
public:
    double score1;
    double score2;
    double score3;
    Student() {
        score1 = 1.23;
        score2 = 3.14;
        score3 = 1.414;
    }
};

class LIBRARY_API StudentHandler {
public:
    uint32_t studentCount;
    Student stu;
    StudentHandler();
    virtual void onStudentIterated(Student stu) = 0;
    uint32_t GetStudentCount();
    virtual ~StudentHandler();
    inline void start(uint32_t iter) {
        for (uint32_t i = 0; i < iter; ++i) {
            onStudentIterated(stu);
        }
    }

};
