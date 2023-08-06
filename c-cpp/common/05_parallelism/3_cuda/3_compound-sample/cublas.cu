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
#include "../cublas_utils.h"

using dtype = float;

__global__ void log_kernel(dtype *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = 0.1 * log(x[i] + 11.0);
  }
}

int main(int argc, char *argv[]) {

  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  int device;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU: " << prop.name << std::endl;

  size_t m = 3000;
  size_t k = 1000;
  size_t n = 2000;
  const size_t lda = m; // ld means "leading dimension"
  const size_t ldb = k;
  const size_t ldc = m;
  const dtype alpha = 0.1;
  const dtype beta = 0.0;

  std::cout << "Reading A..." << std::endl;
  std::vector<dtype> h_A = readVector<dtype>("./a.in", m * k);
  std::cout << "Done (" << h_A.size() << ")\nReading B... " << std::endl;
  std::vector<dtype> h_B = readVector<dtype>("./b.in", k * n);
  std::cout << "Done (" << h_B.size() << ")" << std::endl;
  std::vector<dtype> h_C(m * n, 0.0);

  dtype *d_A = nullptr;
  dtype *d_B = nullptr;
  dtype *d_C = nullptr;

  printf("A\n");
  print_matrix(m, k, h_A.data(), lda);
  printf("=====\n");

  printf("B\n");
  print_matrix(k, n, h_B.data(), ldb);
  printf("=====\n");

  uint64_t t0 = get_timestamp_in_microsec();
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  CUDA_CHECK(cudaMalloc((void **)(&d_A), sizeof(dtype) * m * k));
  CUDA_CHECK(cudaMalloc((void **)(&d_B), sizeof(dtype) * k * n));
  CUDA_CHECK(cudaMalloc((void **)(&d_C), sizeof(dtype) * m * n));

  CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), sizeof(dtype) * m * k,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), sizeof(dtype) * k * n,
                             cudaMemcpyHostToDevice, stream));

  int block_size = 256;
  // Changing it doesn't appear to have a significant impact on the
  // performance of log_kernel<<<num_blocks, block_size>>>(d_A, m * k)--it
  // always takes ~0.15ms
  int num_blocks = (m * k + block_size - 1) / block_size;

  log_kernel<<<num_blocks, block_size>>>(d_A, m * k);

  /* When throwing error, the argument count starts from 0*/
  CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                           d_A, lda, d_B, ldb, &beta, d_C, ldc));

  CUDA_CHECK(cudaMemcpyAsync(h_C.data(), d_C, sizeof(dtype) * m * n,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  uint64_t t1 = get_timestamp_in_microsec();
  std::cout << "C\n";
  print_matrix(m, n, h_C.data(), ldc);
  std::cout << "=====\nWriting C...\n";
  write_matrix_to_csv(h_C, m, n, "./cublas.csv.out");
  std::cout << "Done" << std::endl;

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
