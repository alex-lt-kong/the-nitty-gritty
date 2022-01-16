#include<iostream>
#include<cstring>

using namespace std;

int main(){
    string text = "hello world!";
    text.erase(0, 6);
    cout << text << endl;

}