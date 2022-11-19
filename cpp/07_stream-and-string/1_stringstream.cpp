#include <stdio.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <string.h>

using namespace std;

int main() {
  stringstream ss;
  ss << 100 << ' ' << 200;
  cout << ss.str() << endl;
  const size_t buf_size = 16;
  char buf[buf_size] = {0};
  ss.read(buf, 7);
  printf("[%s]\n", buf);

  ss << "Hello world!";
  ss.read(buf, 6);
  printf("[%s]\n", buf);

  ss.put(0x30);
  ss.put(0x31);
  ss.put(0x33);
  cout << ss.str() << endl;

  memset(buf, 0, buf_size);
  ss.read(buf, 5);
  printf("[%s]\n", buf);

  memset(buf, 0, buf_size);
  ss.read(buf, 4);
  printf("[%s]\n", buf);
  
  return 0;
}