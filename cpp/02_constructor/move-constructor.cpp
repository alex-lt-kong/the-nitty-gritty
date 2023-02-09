#include <string.h>
#include <iostream>
using namespace std;
    
class NaiveString {
private:
    char* _str;
public:
    
    NaiveString(char* str) {
        cout << "[Interal] constructor called" << endl;
        _str = (char*)malloc(sizeof(char) * (strlen(str) + 1));
        if (_str == NULL) {
            throw bad_alloc();
        }
        memcpy(_str, str, strlen(str));
    }
    
    // Copy constructor
    NaiveString(const NaiveString& rhs) {
        cout << "[Interal] copy constructor called, rhs._str: [" << rhs._str
             << "]" << endl;
        _str = (char*)malloc(sizeof(char) * (strlen(rhs._str) + 1));
        if (_str == NULL) {
            throw bad_alloc();
        }
        memcpy(_str, rhs._str, strlen(rhs._str));
    }

    // Move constructor
    NaiveString(NaiveString&& rhs) {
        cout << "[Interal] move constructor called" << endl;
        _str = rhs._str;
        rhs._str = nullptr;
    }


    NaiveString operator+(const NaiveString &rhs) {
        char* temp = (char*)malloc(sizeof(char) *
            (strlen(rhs._str) + strlen(this->_str) + 1));
        if (temp == NULL) {
            throw bad_alloc();
        }
        memcpy(temp, this->_str, strlen(this->_str));
        memcpy(temp + strlen(this->_str), rhs._str, strlen(rhs._str));
        NaiveString ns(temp);
        return ns;
    }

    // Copy assignment operator
    NaiveString operator=(const NaiveString &rhs) {
        
        cout << "[Interal] copy assignment operator called, rhs._str: ["
             << rhs._str << "]" << endl;
        char* temp = (char*)malloc(sizeof(char) * (strlen(rhs._str) + 1));
        if (temp == NULL) {
            throw bad_alloc();
        }
        memcpy(temp, this->_str, strlen(this->_str));
        free(this->_str);
        this->_str = temp;
        return *this;
    }

    // Move assignment operator
    NaiveString operator=(NaiveString &&rhs) {
        
        cout << "[Interal] move assignment operator called, ";
        if (this != &rhs) {
            cout << "and it is making an impact, rhs._str: [" << rhs._str
                 << "]" << endl;
            free(this->_str);
            this->_str = rhs._str;            
            rhs._str = nullptr;
        } else {
            cout << "but it is doing nothing, rhs._str: [" << rhs._str
                 << "]" << endl;
        }
        return *this;
    }


    friend ostream& operator<<(ostream& os, const NaiveString& ns)
    {
        os << ns._str;
        return os;
    }

    ~NaiveString() {
        free(this->_str);
    }
};
    
int main()
{
    cout << "Test 1" << endl;
    NaiveString ns_hel((char*)"Hello ");
    NaiveString ns_hel2 = ns_hel;
    cout << "ns_hel2: " << ns_hel2 << endl << endl;

    cout << "Test 2" << endl;
    NaiveString ns_wor((char*)"world!");
    cout << "ns_hel + ns_wor: " << ns_hel + ns_wor << endl << endl;

    cout << "Test 3" << endl;
    NaiveString ns_hw0 = ns_hel + ns_wor;
    // This mostly triggers copy constructor with copy elision
    cout << "ns_hw0: " << ns_hw0 << endl << endl; 

    cout << "Test 4" << endl;
    NaiveString ns_hw1 = NaiveString((char*)"Hello world");
    cout << "ns_hw1: " << ns_hw1 << endl;
    ns_hw1 = ns_hw1;
    cout << "ns_hw1: " << ns_hw1 << endl;
    ns_hw1 = NaiveString((char*)"Goodbye");    
    cout << "ns_hw1: " << ns_hw1 << endl << endl;

    cout << "Test 5" << endl;
    NaiveString ns_fb = NaiveString((char*)"foobar");
    cout << "ns_fb: " << ns_fb << endl;
    ns_fb = ns_hw1;
    cout << "ns_fb: " << ns_fb << endl << endl;


    return 0;
}