#include <iostream>
#include <random>

#include "string-demo.h"

using namespace std;
/*
void ssoNaiveDemo() {
    string shortString = "foobar"; // 0x66, 0x6f, 0x6f, 0x62, 0x61, 0x72, 0x00
    shortString += to_string(rand() % 10);
    cout << "shortString: " << shortString << endl;
}*/

void longStringNoSsoDemo(string prefix) {
    string longString = "Hello, world! This string is longer than 11/23 characters.";
    longString += prefix;
    cout << "longString: " << longString << endl;
}