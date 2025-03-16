#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

std::string __attribute__((noinline)) gen_random(const int len) {
  static const char alphanum[] = "0123456789"
                                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyz";
  std::string tmp_s;
  tmp_s.reserve(len);

  for (int i = 0; i < len; ++i) {
    tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
  }

  return tmp_s;
}

vector<string> __attribute__((noinline)) getStringVec() {
  vector<string> strvec;
  for (int i = 0; i < 10; ++i) {
    string str = gen_random(i);
    strvec.push_back(str);
  }
  return strvec;
}

int main() {
  vector<string> strvec;
  for (int i = 0; i < 10; ++i) {
    string str = gen_random(i);
    strvec.push_back(str);
  }
  for (int i = 0; i < 10; ++i) {
    cout << strvec[i] << endl;
  }
}
