#include <iostream>

using namespace std;

class DatabaseHelper {
public:
DatabaseHelper() {
    cout << "DatabaseHelper() called" << endl;
}
~DatabaseHelper() {
    cout << "~DatabaseHelper() called" << endl;
}
void aFancyMethod() {
    cout << "aFancyMethod() called" << endl;
}

void anInvalidMethod() {
    cout << "anInvalidMethod() called, exception will be thrown" << endl;
    throw "A snake oil exception!";
}
};


int main() {
    try {
        DatabaseHelper myHelper = DatabaseHelper();
        myHelper.aFancyMethod();
        myHelper.anInvalidMethod();
    } catch(const char* msg) {
        cout << "A char exception caught, its content is: " << msg << endl;
    }
    cout << "Fear not, ~DatabaseHelper() will be called anyway!" << endl;
    return 0;
}
