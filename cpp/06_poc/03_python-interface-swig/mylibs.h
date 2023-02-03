using namespace std;

typedef struct TranscriptStruct{
    double Scores[5];
} Transcript;


class MyClass {

public:
    int Id;
    char* Name;
    unsigned int PhoneNumber;
    Transcript Scores;

    MyClass();
    ~MyClass();
    void Print();
    void PrintOther(MyClass mc);
};

class Student {
public:
    double score1;
    double score2;
    double score3;
    Student() {
        score1 = 0.0;
        score2 = 0.0;
        score3 = 0.0;
    }
};

class MyIf {
public:
    uint32_t count;
    Student stu;
    inline MyIf() {
        count = 0;
        stu = Student();
    }
    virtual void myfunc(Student a) {};
    inline void start(uint32_t iter) {
        for (uint32_t i = 0; i < iter; ++i) {
            myfunc(stu);
        }
    }

};
