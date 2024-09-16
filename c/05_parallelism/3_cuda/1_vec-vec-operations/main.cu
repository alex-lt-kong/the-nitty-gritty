#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "../../utils.h"

typedef void calculationRoutine(const float *a, const float *b, float *c,
                                const ssize_t len);

/* The function name/signature doesn't matter, it will be transparently sent to
 * GPU to execute.
 * This function, to be executed on NVIDIA GPU, is also known as "CUDA kernel".
 */
__global__ void gpuVectorAdd(const float *a, const float *b, float *c,
                             const ssize_t len) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) {
    c[i] = a[i] + b[i];
  }
}

__global__ void gpuVectorMul(const float *a, const float *b, float *c,
                             const ssize_t len) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) {
    c[i] = a[i] * b[i];
  }
}

__global__ void gpuVectorDiv(const float *a, const float *b, float *c,
                             const ssize_t len) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) {
    c[i] = a[i] / b[i];
  }
}

__global__ void gpuVectorPow(const float *a, const float *b, float *c,
                             const ssize_t len) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) {
    c[i] = pow(a[i], b[i]);
  }
}

void cpuVectorAdd(const float *a, const float *b, float *c, const ssize_t len) {
  for (int i = 0; i < len; ++i) {
    c[i] = a[i] + b[i];
  }
}

void cpuVectorMul(const float *a, const float *b, float *c, const ssize_t len) {
  for (int i = 0; i < len; ++i) {
    c[i] = a[i] * b[i];
  }
}

void cpuVectorDiv(const float *a, const float *b, float *c, const ssize_t len) {
  for (int i = 0; i < len; ++i) {
    c[i] = a[i] / b[i];
  }
}

void cpuVectorPow(const float *a, const float *b, float *c, const ssize_t len) {
  for (int i = 0; i < len; ++i) {
    c[i] = pow(a[i], b[i]);
  }
}

void callCPURoutine(calculationRoutine funcPtr, float *a, float *b, float *c,
                    const ssize_t len) {
  uint64_t t0, t1;
  uint64_t cpu_elapsed;
  printf("--- Running on CPU ---\n");
  t0 = get_timestamp_in_microsec();
  funcPtr(a, b, c, len);
  t1 = get_timestamp_in_microsec();
  cpu_elapsed = t1 - t0;
  printf("Done, took %.2lfms\n", cpu_elapsed / 1000.0);
}

void callGPURoutine(calculationRoutine funcPtr, float *a, float *b, float *c,
                    const ssize_t len) {

  cudaError_t cudaError;
  uint64_t t0, t1, elapsed, T0, T1;

  printf("--- Running on GPU ---\n");
  T0 = get_timestamp_in_microsec();
  float *cudaA = NULL;
  float *cudaB = NULL;
  float *cudaC = NULL;
  int threadsPerBlock, blocksPerGrid;
  // Allocate memory for pointers into the GPU
  if ((cudaError = cudaMalloc(&cudaA, sizeof(float) * len)) != cudaSuccess ||
      (cudaError = cudaMalloc(&cudaB, sizeof(float) * len)) != cudaSuccess ||
      (cudaError = cudaMalloc(&cudaC, sizeof(float) * len)) != cudaSuccess) {
    // C/C++ implements short-circuit evaluation, meaning that for the ||
    // operator, if the first argument is evaluted to true, the 2nd
    // argument will not be evaluted
    fprintf(stderr, "cudaMalloc() failed: %s\n", cudaGetErrorString(cudaError));
    goto err_cuda_malloc;
  }

  t0 = get_timestamp_in_microsec();
  // Copy vectors into the GPU
  if ((cudaError = cudaMemcpy(cudaA, a, len * sizeof(float),
                              cudaMemcpyHostToDevice)) != cudaSuccess ||
      (cudaError = cudaMemcpy(cudaB, b, len * sizeof(float),
                              cudaMemcpyHostToDevice)) != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed: %s\n", cudaGetErrorString(cudaError));
    goto err_cuda_memcpy;
  }
  cudaDeviceSynchronize();
  t1 = get_timestamp_in_microsec();
  elapsed = t1 - t0;
  printf("Took %.2lfms to move data from RAM to GPU memory (%.1lfMB/sec)\n",
         elapsed / 1000.0,
         2 * len * sizeof(float) / 1024.0 / 1024 / (elapsed / 1000.0 / 1000.0));

  threadsPerBlock = 128;
  blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  t0 = get_timestamp_in_microsec();
  funcPtr<<<blocksPerGrid, threadsPerBlock>>>(cudaA, cudaB, cudaC, len);
  cudaDeviceSynchronize();
  t1 = get_timestamp_in_microsec();
  printf("Took %.2lfms to calculate\n", (t1 - t0) / 1000.0);

  t0 = get_timestamp_in_microsec();
  if ((cudaError = cudaMemcpy(c, cudaC, len * sizeof(float),
                              cudaMemcpyDeviceToHost)) != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed: %s\n", cudaGetErrorString(cudaError));
    goto err_cuda_memcpy;
  }
  t1 = get_timestamp_in_microsec();
  elapsed = t1 - t0;
  printf("Took %.2lfms to move data from GPU memory to RAM (%.1lfMB/sec)\n",
         elapsed / 1000.0,
         len * sizeof(float) / 1024.0 / 1024 / (elapsed / 1000.0 / 1000.0));
err_cuda_memcpy:
err_cuda_malloc:
  cudaFree(cudaA);
  cudaFree(cudaB);
  cudaFree(cudaC);
  T1 = get_timestamp_in_microsec();
  printf("Done, took %.2lfms\n", (T1 - T0) / 1000.0);
}

void prepareRandomNumbers(float *a, float *b, ssize_t len) {
  for (int i = 0; i < len; ++i) {
    a[i] = rand() % (RAND_MAX / 2 - 1);
    b[i] = rand() % (RAND_MAX / 2 - 1);
    a[i] /= (float)RAND_MAX;
    b[i] /= (float)RAND_MAX;
    if (fabs(a[i]) < 0.01) {
      a[i] += 1;
    }
    if (fabs(b[i]) < 0.01) {
      b[i] += 1;
    }
  }
  printf("%.1lf MB random data generated\n",
         2 * len * sizeof(float) / 1024.0 / 1024);
}

void checkResults(const float *c_cpu, const float *c_gpu, const ssize_t len) {
  printf("\nChecking if CPU/GPU results are identical...");

  int inconsistent_count = 0;
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
    printf("YES!\n\n");
  } else {
    fprintf(stderr, "%d mismatches found!\n\n", inconsistent_count);
  }
}

void printCPUandGPU() {
  FILE *fp = fopen("/proc/cpuinfo", "r");
  char line[PATH_MAX];
  char *version = NULL;

  while (fgets(line, PATH_MAX, fp)) {
    if (strstr(line, "model name") != NULL) {
      version = strchr(line, ':') + 2; // skip over ": "
      break;
    }
  }

  fclose(fp);

  if (version) {
    printf("CPU: %s", version);
  } else {
    printf("Failed to get CPU version.");
  }

  int device;
  cudaDeviceProp prop;
  cudaError_t cudaError;
  if ((cudaError = cudaGetDevice(&device)) != cudaSuccess ||
      (cudaError = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) {
    fprintf(stderr, "cudaGetDevice() failed: %s\n",
            cudaGetErrorString(cudaError));
  }
  printf("GPU: %s\n\n", prop.name);
}

int main(void) {
  int retval = 0;
  printCPUandGPU();
  const ssize_t len = 200 * 1000 * 1000;
  float *a = (float *)malloc(len * sizeof(float));
  float *b = (float *)malloc(len * sizeof(float));
  float *cCPU = (float *)calloc(len, sizeof(float));
  float *cGPU = (float *)calloc(len, sizeof(float));

  srand(time(NULL));

  calculationRoutine *cpuFuncPtrs[] = {&cpuVectorAdd, &cpuVectorMul,
                                       &cpuVectorDiv, &cpuVectorPow};
  calculationRoutine *gpuFuncPtrs[] = {&gpuVectorAdd, &gpuVectorMul,
                                       &gpuVectorDiv, &gpuVectorPow};
  char routineNames[][32] = {"vectorAdd", "vectorMul", "vectorDiv",
                             "vectorPow"};

  if (a == NULL || b == NULL || cCPU == NULL || cGPU == NULL) {
    fprintf(stderr, "malloc() failed\n");
    goto err_malloc;
  }
  prepareRandomNumbers(a, b, len);

  for (int i = 0; i < sizeof(cpuFuncPtrs) / sizeof(calculationRoutine *); ++i) {
    printf("\n========== Now running: %s ==========\n", routineNames[i]);
    (void)callCPURoutine(cpuFuncPtrs[i], a, b, cCPU, len);
    (void)callGPURoutine(gpuFuncPtrs[i], a, b, cGPU, len);
    checkResults(cCPU, cGPU, len);
  }

err_malloc:
  free(a);
  free(b);
  free(cCPU);
  free(cGPU);
  return retval;
}