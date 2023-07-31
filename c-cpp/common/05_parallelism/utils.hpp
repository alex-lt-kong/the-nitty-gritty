#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
void write_matrix_to_csv(const std::vector<T> &vec, const int m, const int n,
                         const std::string &path) {
  if (vec.size() != m * n) {
    throw std::runtime_error("vector size does not match matrix dimensions");
  }

  // Open the output file stream
  std::ofstream file(path);

  // Write the matrix elements to the file
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      file << vec[i + j * m];
      if (j < n - 1) {
        file << ",";
      }
    }
    file << std::endl;
  }

  // Close the file stream
  file.close();
}

template <typename T>
void print_matrix(const int m, const int n, const T *A, const int lda) {
  std::cout << "[\n";
  for (int i = 0; i < m; i++) {
    if (m < 20 || (i < 5 || i > m - 5)) {
      for (int j = 0; j < n; j++) {

        if (n < 20 || (j < 5 || j > n - 5)) {
          if (j == 0) {
            std::cout << "  [";
          }

          std::cout << A[j * lda + i];

          if (j == n - 1) {
            std::cout << "],\n";
          } else {
            std::cout << ", ";
          }
        } else if (j == 5) {
          std::cout << " ... ";

        } else {
          continue;
        }
      }
    } else if (i == 5) {
      std::cout << "...\n";
    } else {
      continue;
    }
  }
  std::cout << "]" << std::endl;
}

template <typename T>
std::vector<T> readVector(std::string filePath, size_t numRead) {
  std::ifstream infile(filePath);
  if (!infile.is_open()) {
    throw std::runtime_error(filePath + " can't be opened");
  }
  std::vector<T> vec;
  T value;
  size_t actualRead = 0;
  while ((infile >> value) && (actualRead < numRead)) {
    vec.push_back(value);
    ++actualRead;
  }
  // infile.close(); Leave it to RAII!
  if (actualRead < numRead) {
    throw std::runtime_error(
        filePath + "'s EOF reached before " + std::to_string(numRead) +
        " values are read (read " + std::to_string(actualRead) + ")");
  }
  return vec;
}
