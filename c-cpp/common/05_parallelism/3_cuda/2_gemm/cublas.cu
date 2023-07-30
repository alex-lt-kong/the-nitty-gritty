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

int main(int argc, char *argv[]) {
  int device;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  cout << "GPU: " << prop.name << endl;
  cublasHandle_t cublasH = NULL;
  // cudaStream_t stream = NULL;

  size_t m = 10000;
  size_t k = 4000;
  size_t n = 6000;
  const size_t lda = m; // ld means "leading dimension"
  const size_t ldb = k;
  const size_t ldc = m;

  cout << "Reading A..." << endl;
  vector<dtype> h_A = readVector<dtype>("./a.in", m * k);
  cout << "Done\nReading B..." << endl;
  vector<dtype> h_B = readVector<dtype>("./b.in", k * n);
  cout << "Done" << endl;

  dtype *h_C;
  CUDA_CHECK(cudaMallocHost((void **)(&h_C), m * n * sizeof(dtype)));
  const dtype alpha = 0.1;
  const dtype beta = 0.0;

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
  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));

  // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // CUBLAS_CHECK(cublasSetStream(cublasH, stream));
  uint64_t t1 = get_timestamp_in_microsec();
  /* step 2: copy data to device */
  CUDA_CHECK(cudaMalloc((void **)(&d_A), sizeof(dtype) * m * k));
  CUDA_CHECK(cudaMalloc((void **)(&d_B), sizeof(dtype) * k * n));
  CUDA_CHECK(cudaMalloc((void **)(&d_C), sizeof(dtype) * m * n));
  uint64_t t2 = get_timestamp_in_microsec();
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(dtype) * m * k,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(dtype) * k * n,
                        cudaMemcpyHostToDevice));

  cudaDeviceSynchronize();
  uint64_t t3 = get_timestamp_in_microsec();
  /* When throwing error, the argument count starts from 0*/
  CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                           d_A, lda, d_B, ldb, &beta, d_C, ldc));
  cudaDeviceSynchronize();
  uint64_t t4 = get_timestamp_in_microsec();
  /* step 4: copy data to host */
  CUDA_CHECK(
      cudaMemcpy(h_C, d_C, sizeof(dtype) * m * n, cudaMemcpyDeviceToHost));
  // CUDA_CHECK(cudaStreamSynchronize(stream));
  uint64_t t5 = get_timestamp_in_microsec();

  printf("C\n");
  // print_matrix(m, n, h_C, ldc);
  printf("=====\n");

  /* free resources */
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUBLAS_CHECK(cublasDestroy(cublasH));
  // CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceReset());
  cout << "CUDA Init: " << (t1 - t0) / 1000.0 << "ms" << endl
       << "cudaMalloc(): " << (t2 - t1) / 1000.0 << "ms" << endl
       << "cudaMemcpy(HostToDevice): " << (t3 - t2) / 1000.0 << "ms" << endl
       << "cublasDgemm(): " << (t4 - t3) / 1000.0 << "ms" << endl
       << "cudaMemcpy(DeviceToHost): " << (t5 - t4) / 1000.0 << "ms" << endl
       << "Total: " << (t5 - t0) / 1000.0 << "ms" << endl;
  return EXIT_SUCCESS;
}
