#include<iostream>
#include<cstring>

using namespace std;

int main(){

    cout << "Content-type:text/html\n\n";
    if(strcmp(getenv("REQUEST_METHOD"),"GET")==0){

        cout << "<html>Your dynamic query string is:<br>";
        cout << getenv("QUERY_STRING") << endl;
        cout << "</html>";
    } else {
        cout << "Method [" << getenv("REQUEST_METHOD") << "] not supported";
    }
    return 0;
}
