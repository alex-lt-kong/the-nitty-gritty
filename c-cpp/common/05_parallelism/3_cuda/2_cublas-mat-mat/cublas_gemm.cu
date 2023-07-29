// Originally
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Level-3/gemm
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

int main(int argc, char *argv[]) {
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  const int m = 4;
  const int n = 3;
  const int k = n;
  int lda;
  int ldb;
  int ldc = n;
  /*
   *   A = | 1.0 | 2.0 |
   *       | 3.0 | 4.0 |
   *
   *   B = | 5.0 | 6.0 |
   *       | 7.0 | 8.0 |
   */

  double *A;
  generate_random_matrix(m, n, &A, &lda);
  double *B;
  generate_random_matrix(n, m, &B, &ldb);
  printf("lda: %d, ldb: %d\n", lda, ldb);
  double *C = (double *)malloc(sizeof(double) * m * m);
  const double alpha = 1.0;
  const double beta = 0.0;

  double *d_A = nullptr;
  double *d_B = nullptr;
  double *d_C = nullptr;

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  printf("A\n");
  print_matrix(m, n, A, lda);
  printf("=====\n");

  printf("B\n");
  print_matrix(n, m, B, ldb);
  printf("=====\n");

  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  /* step 2: copy data to device */
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * m * n));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * n * m));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * m * m));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeof(double) * m * n,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeof(double) * n * m,
                             cudaMemcpyHostToDevice, stream));

  /* step 3: compute */
  /* When throwing error, the argument count starts from 0*/
  CUBLAS_CHECK(cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda,
                           d_B, ldb, &beta, d_C, m));

  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeof(double) * m * m,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /*
   *   C = | 23.0 | 31.0 |
   *       | 34.0 | 46.0 |
   */

  printf("C\n");
  print_matrix(m, m, C, ldc);
  printf("=====\n");

  /* free resources */
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}
