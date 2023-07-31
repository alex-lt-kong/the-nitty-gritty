#include <cblas.h>
#include <stdio.h>

#include "../../utils.h"
#include "../../utils.hpp"

using dtype = float;

int main(void) {
  blasint m = 30000;
  blasint k = 8000;
  blasint n = 1100;
  const blasint lda = m;
  const blasint ldb = k;
  const blasint ldc = m;
  const dtype alpha = 0.1;
  const dtype beta = 0.0;
  std::cout << "Reading A..." << std::endl;
  std::vector<dtype> A = readVector<dtype>("./a.in", m * k);
  std::cout << "Done (" << A.size() << ")\nReading B... " << std::endl;
  std::vector<dtype> B = readVector<dtype>("./b.in", k * n);
  std::cout << "Done (" << B.size() << ")" << std::endl;
  std::vector<dtype> C(m * n);

  printf("A\n");
  print_matrix(m, k, A.data(), lda);
  printf("=====\n");

  printf("B\n");
  print_matrix(k, n, B.data(), ldb);
  std::cout << "=====\n";

  std::cout << "Done" << std::endl;
  uint64_t t0 = get_timestamp_in_microsec();
  /* When throwing error, the argument count starts from 0*/
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
  uint64_t t1 = get_timestamp_in_microsec();

  print_matrix(m, n, C.data(), m);
  std::cout << "=====\nWriting C...\n";
  write_matrix_to_csv(C, m, n, "./openblas.csv.out");
  std::cout << "Done" << std::endl;
  std::cout << "Total: " << (t1 - t0) / 1000.0 << "ms" << std::endl;
  return 0;
}