// Originally
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Level-3/gemm
// Doc: https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../../utils.h"
#include "../../utils.hpp"
#include "cublas_utils.h"

__global__ void log_kernel(double *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = 0.1 * log(x[i] + 11.0);
  }
}

int main(int argc, char *argv[]) {
  int device;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  cout << "GPU: " << prop.name << endl;
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  size_t m = 100 * 1000;
  size_t k = 1999;
  size_t n = 30;
  const size_t lda = m; // ld means "leading dimension"
  const size_t ldb = k;
  const size_t ldc = m;

  cout << "Reading A..." << endl;
  vector<double> A = readDoubleVector("./a.in", m * k);
  cout << "Done\nReading B..." << endl;
  vector<double> B = readDoubleVector("./b.in", k * n);
  cout << "Done" << endl;
  cout << B.size() << endl;
  vector<double> C(m * n, 0.0);
  const double alpha = 1.0;
  const double beta = 0.0;

  double *d_A = nullptr;
  double *d_B = nullptr;
  double *d_C = nullptr;

  printf("A\n");
  print_matrix(m, k, A.data(), lda);
  printf("=====\n");

  printf("B\n");
  print_matrix(k, n, B.data(), ldb);
  printf("=====\n");

  uint64_t t0 = get_timestamp_in_microsec();
  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  /* step 2: copy data to device */
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * m * k));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * k * n));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * m * n));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * m * k,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(double) * k * n,
                             cudaMemcpyHostToDevice, stream));
  int block_size = 256;
  int num_blocks = (m * k + block_size - 1) / block_size;

  CUDA_CHECK(cudaStreamSynchronize(stream));
  log_kernel<<<num_blocks, block_size>>>(d_A, m * k);

  /* When throwing error, the argument count starts from 0*/
  CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                           d_A, lda, d_B, ldb, &beta, d_C, ldc));

  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(double) * m * n,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  uint64_t t1 = get_timestamp_in_microsec();
  printf("C\n");
  print_matrix(m, n, C.data(), ldc);
  printf("=====\n");

  /* free resources */
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());
  printf("%.02fms\n", (t1 - t0) / 1000.0);
  return EXIT_SUCCESS;
}
