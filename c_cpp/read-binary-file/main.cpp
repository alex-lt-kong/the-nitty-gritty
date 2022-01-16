#include<iostream>
#include<cstring>
#include <ctime>
#include <fstream>

using namespace std;

int main(){
    
    ifstream in("main.in", ios::binary | ios::ate);
    //size_t size = in.tellg();
    size_t size = 10;
    char *buffer = new char[size];
    int pos = 0;
    while (in.eof() == false) {
      in.seekg(pos, ios::beg);
      in.read(buffer, size);
      cout << buffer;
      pos += size;
    }
}