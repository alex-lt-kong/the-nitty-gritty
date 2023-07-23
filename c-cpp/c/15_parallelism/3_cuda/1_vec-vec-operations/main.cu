#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "../../utils.h"

typedef void calculationRoutine(double *a, double *b, double *c, ssize_t len);

/* The function name/signature doesn't matter, it will be transparently sent to
 * GPU to execute.
 * This function, to be executed on NVIDIA GPU, is also known as "CUDA kernel".
 */
__global__ void gpuVectorAddition(double *a, double *b, double *c,
                                  ssize_t len) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) {
    c[i] = sqrt(a[i] / b[i]) + sqrt(b[i] / a[i]) + pow(a[i], 0.33) +
           pow(b[i], 0.22);
  }
}

void cpuVectorAddition(const double *a, const double *b, double *c,
                       const ssize_t len) {
  for (int i = 0; i < len; ++i) {
    c[i] = sqrt(a[i] / b[i]) + sqrt(b[i] / a[i]) + pow(a[i], 0.33) +
           pow(b[i], 0.22);
  }
}

void gpu_version(calculationRoutine funcPtr, double *a, double *b, double *c,
                 const ssize_t len) {

  double *cuda_a = NULL;
  double *cuda_b = NULL;
  double *cuda_c = NULL;
  int threads_per_block, blocks_per_grid;
  cudaError_t cu_error;
  // Allocate memory for pointers into the GPU
  if ((cu_error = cudaMalloc(&cuda_a, sizeof(double) * len)) != cudaSuccess ||
      (cu_error = cudaMalloc(&cuda_b, sizeof(double) * len)) != cudaSuccess ||
      (cu_error = cudaMalloc(&cuda_c, sizeof(double) * len)) != cudaSuccess) {
    // C/C++ implements short-circuit evaluation, meaning that for the ||
    // operator, if the first argument is evaluted to true, the 2nd
    // argument will not be evaluted
    fprintf(stderr, "cudaMalloc() failed: %s\n", cudaGetErrorString(cu_error));
    goto err_cuda_malloc;
  }

  uint64_t t0, t1, elapsed;

  t0 = get_timestamp_in_microsec();
  // Copy vectors into the GPU
  if ((cu_error = cudaMemcpy(cuda_a, a, len * sizeof(double),
                             cudaMemcpyHostToDevice)) != cudaSuccess ||
      (cu_error = cudaMemcpy(cuda_b, b, len * sizeof(double),
                             cudaMemcpyHostToDevice)) != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed: %s\n", cudaGetErrorString(cu_error));
    goto err_cuda_memcpy;
  }
  t1 = get_timestamp_in_microsec();
  elapsed = t1 - t0;
  printf("Took %.2lfms to move data from RAM to GPU memory (%.1lfMB/sec)\n",
         elapsed / 1000.0,
         2 * len * sizeof(double) / 1024.0 / 1024 /
             (elapsed / 1000.0 / 1000.0));

  threads_per_block = 128;
  blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;
  t0 = get_timestamp_in_microsec();
  funcPtr<<<blocks_per_grid, threads_per_block>>>(cuda_a, cuda_b, cuda_c, len);
  t1 = get_timestamp_in_microsec();
  printf("Took %.2lfms to calculate\n", (t1 - t0) / 1000.0);

  t0 = get_timestamp_in_microsec();
  if ((cu_error = cudaMemcpy(c, cuda_c, len * sizeof(double),
                             cudaMemcpyDeviceToHost)) != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed: %s\n", cudaGetErrorString(cu_error));
    goto err_cuda_memcpy;
  }
  t1 = get_timestamp_in_microsec();
  elapsed = t1 - t0;
  printf("Took %.2lfms to move data from GPU memory to RAM (%.1lfMB/sec)\n",
         elapsed / 1000.0,
         len * sizeof(double) / 1024.0 / 1024 / (elapsed / 1000.0 / 1000.0));
err_cuda_memcpy:
err_cuda_malloc:
  cudaFree(cuda_a);
  cudaFree(cuda_b);
  cudaFree(cuda_c);
}

int main(void) {
  int retval = 0;
  int device;
  cudaDeviceProp prop;
  int inconsistent_count = 0;

  srand(time(NULL));
  const ssize_t len = 200 * 1000 * 1000;
  double *a = (double *)malloc(len * sizeof(double));
  double *b = (double *)malloc(len * sizeof(double));
  double *c_cpu = (double *)calloc(len, sizeof(double));
  double *c_gpu = (double *)calloc(len, sizeof(double));

  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    retval = -1;
    fprintf(stderr, "GPU error\n");
    goto err_gpu_error;
  }
  if (a == NULL || b == NULL || c_cpu == NULL || c_gpu == NULL) {
    fprintf(stderr, "malloc() failed\n");
    goto err_malloc;
  }

  for (int i = 0; i < len; ++i) {
    a[i] = rand() % (RAND_MAX / 2 - 1);
    b[i] = rand() % (RAND_MAX / 2 - 1);
    a[i] /= (double)RAND_MAX;
    b[i] /= (double)RAND_MAX;
    if (fabs(a[i]) < 0.01) {
      a[i] += 1;
    }
    if (fabs(b[i]) < 0.01) {
      b[i] += 1;
    }
  }
  printf("%.1lf MB random data generated\n",
         2 * len * sizeof(double) / 1024.0 / 1024);

  usleep(1000); // sleep for 1 ms
  uint64_t t0, t1;
  uint64_t cpu_elapsed, gpu_elapsed;

  printf("=== Running on CPU ===\n");
  t0 = get_timestamp_in_microsec();
  (void)cpuVectorAddition(a, b, c_cpu, len);
  t1 = get_timestamp_in_microsec();
  cpu_elapsed = t1 - t0;
  printf("Done, took %.2lfms\n\n", cpu_elapsed / 1000.0);

  printf("=== Running on GPU (%s) ===\n", prop.name);
  t0 = get_timestamp_in_microsec();
  (void)gpu_version(&gpuVectorAddition, a, b, c_gpu, len);
  t1 = get_timestamp_in_microsec();
  gpu_elapsed = t1 - t0;
  printf("Done, took %.2lfms\n\n", gpu_elapsed / 1000.0);

  printf("Checking if CPU/GPU results are identical...\n");

  for (int i = 0; i < len; ++i) {
    if (fabs(c_cpu[i] - c_gpu[i]) > 1e-6) {
      fprintf(stderr, "%d-th element is DIFFERENT (%lf vs %lf)!!!\n", i,
              c_cpu[i], c_gpu[i]);
      ++inconsistent_count;
    }
    if (inconsistent_count >= 10) {
      fprintf(stderr, "too many mismatches, check aborted\n");
      break;
    }
  }
  if (inconsistent_count == 0) {
    printf("YES!\n");
  } else {
    fprintf(stderr, "%d mismatches found!\n", inconsistent_count);
  }

err_malloc:
  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
err_gpu_error:
  return retval;
}