#include <fstream>
#include <string>
#include <vector>

using namespace std;

inline vector<double> readDoubleVector(string filePath, size_t numRead) {
  ifstream infile(filePath);
  if (!infile.is_open()) {
    throw runtime_error(filePath + " can't be opened");
  }
  vector<double> dblVec;
  double value;
  size_t actualRead = 0;
  while ((infile >> value) && (actualRead <= numRead)) {
    dblVec.push_back(value);
    ++actualRead;
  }
  // infile.close(); Leave it to RAII!
  if (actualRead < numRead) {
    throw runtime_error(filePath + "'s EOF reached before " +
                        to_string(numRead) + " doubles are read (read " +
                        to_string(actualRead) + ")");
  }
  return dblVec;
}
