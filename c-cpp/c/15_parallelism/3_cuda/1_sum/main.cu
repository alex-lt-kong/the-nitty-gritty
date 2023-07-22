#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>

__global__ void vectorAdd(int *a, int* b, int* c)  {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

uint64_t get_timestamp_in_microsec() {
  struct timeval tv;
  gettimeofday(&tv, NULL); 
  return 1000000 * tv.tv_sec + tv.tv_usec;
}

void cpu_version(const int* a, const int* b, int* c, const ssize_t len) {
  for (int i = 0; i < len; ++i) {
    c[i] = a[i] + b[i];
  }
}

void gpu_version(int* a, int* b, int* c, const ssize_t len) {

  int* cuda_a = 0;
  int* cuda_b = 0;
  int* cuda_c = 0;
  int threads_per_block, blocks_per_grid;

  // Allocate memory for pointers into the GPU
  if (cudaMalloc(&cuda_a, sizeof(int) * len) != cudaSuccess || 
      cudaMalloc(&cuda_b, sizeof(int) * len) != cudaSuccess || 
      cudaMalloc(&cuda_c, sizeof(int) * len) != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed\n");
    goto err_cuda_malloc;
  }
  
  uint64_t t0, t1, elapsed;

  t0 = get_timestamp_in_microsec();
  // Copy vectors into the GPU
  if (cudaMemcpy(cuda_a, a, len * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess || 
      cudaMemcpy(cuda_b, b, len * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed\n");
    goto err_cuda_memcpy;
  }
  t1 = get_timestamp_in_microsec();
  elapsed = t1 - t0;
  printf("Took %.2lfms to move data from RAM to GPU memory (%.1lfMB/sec)\n",
    elapsed / 1000.0, 2 * len * sizeof(int) / 1024.0 / 1024 / (elapsed / 1000.0 / 1000.0));


  threads_per_block = 64;
  blocks_per_grid =(len + threads_per_block - 1) / threads_per_block;
  t0 = get_timestamp_in_microsec();
  vectorAdd<<<blocks_per_grid, threads_per_block>>>(cuda_a, cuda_b, cuda_c);
  t1 = get_timestamp_in_microsec();
  printf("Took %.2lfms to calculate\n", (t1 - t0) / 1000.0);

  t0 = get_timestamp_in_microsec();
  if (cudaMemcpy(c, cuda_c, len * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed\n");
    goto err_cuda_memcpy;
  }
  t1 = get_timestamp_in_microsec();
  elapsed = t1 - t0;
  printf("Took %.2lfms to move data from GPU memory to RAM (%.1lfMB/sec)\n",
    elapsed / 1000.0, len * sizeof(int) / 1024.0 / 1024 / (elapsed / 1000.0 / 1000.0));
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
  const ssize_t len = 1000 * 1000 * 1000;
  int* a = (int*)malloc(len * sizeof(int));
  int* b = (int*)malloc(len * sizeof(int));
  int* c_cpu = (int*)calloc(len, sizeof(int));
  int* c_gpu = (int*)calloc(len, sizeof(int));

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
    a[i] = rand() % (RAND_MAX/2-1);
    b[i] = rand() % (RAND_MAX/2-1);
  }
  usleep(5 * 1000); // sleep for 500 ms (500000 microseconds)
  uint64_t t0, t1;
  uint64_t cpu_elapsed, gpu_elapsed;

  printf("=== Running on CPU ===\n");
  t0 = get_timestamp_in_microsec();
  (void)cpu_version(a, b, c_cpu, len);
  t1 = get_timestamp_in_microsec();
  cpu_elapsed = t1 - t0;
  printf("Done, took %.2lfms\n\n", cpu_elapsed / 1000.0);

  printf("=== Running on GPU (%s) ===\n", prop.name);
  t0 = get_timestamp_in_microsec();
  (void)gpu_version(a, b, c_gpu, len);
  t1 = get_timestamp_in_microsec();
  gpu_elapsed = t1 - t0;
  printf("Done, took %.2lfms\n\n", gpu_elapsed / 1000.0);
  
  printf("Checking if CPU/GPU results are identical...\n");
  
  for (int i = 0; i < len; ++i) {
    if (c_cpu[i] != c_gpu[i]) {
      fprintf(stderr, "%d-th element is DIFFERENT!!!\n", i);
      ++inconsistent_count;
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