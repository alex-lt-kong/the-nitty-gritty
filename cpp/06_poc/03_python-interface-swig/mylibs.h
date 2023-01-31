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
};
