#include <fstream>
#include <string>
#include <vector>

using namespace std;

template <typename T> vector<T> readVector(string filePath, size_t numRead) {
  ifstream infile(filePath);
  if (!infile.is_open()) {
    throw runtime_error(filePath + " can't be opened");
  }
  vector<T> vec;
  T value;
  size_t actualRead = 0;
  while ((infile >> value) && (actualRead <= numRead)) {
    vec.push_back(value);
    ++actualRead;
  }
  // infile.close(); Leave it to RAII!
  if (actualRead < numRead) {
    throw runtime_error(filePath + "'s EOF reached before " +
                        to_string(numRead) + " values are read (read " +
                        to_string(actualRead) + ")");
  }
  return vec;
}
