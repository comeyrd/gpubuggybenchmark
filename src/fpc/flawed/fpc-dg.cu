#include <stdio.h>      /* defines printf for tests */
#include <stdlib.h> 
#include <chrono>
#include <cuda.h>
#include "fpc-dg.hpp"
#include "cuda-utils.hpp"
//Data-locality - Using global for shared memory
//TODO

namespace {
__device__
unsigned my_abs ( int x )
{
  unsigned t = x >> 31;
  return (x ^ t) - t;
}
}

__global__ void
fpc_dg_kernel(const ulong *values, unsigned *cmp_size, size_t* length,unsigned* compressable) {
    int lid = threadIdx.x;
    int WGS = blockDim.x;
    int block_ix = blockIdx.x;
    size_t gid = block_ix * WGS + lid;
    if(gid > *length) return;
    ulong value = values[gid];
    unsigned inc;

    // 000
    if (value == 0) {
        inc = 1;
    }
    // 001 010
    else if ((my_abs((int)(value)) <= 0xFF)) {
        inc = 1;
    }
    // 011
    else if ((my_abs((int)(value)) <= 0xFFFF)) {
        inc = 2;
    }
    // 100
    else if ((((value) & 0xFFFF) == 0)) {
        inc = 2;
    }
    // 101
    else if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF && my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) {
        inc = 2;
    }
    // 110
    else if ((((value) & 0xFF) == ((value >> 8) & 0xFF)) &&
             (((value) & 0xFF) == ((value >> 16) & 0xFF)) &&
             (((value) & 0xFF) == ((value >> 24) & 0xFF))) {
        inc = 1;
    } else {
        inc = 4;
    }

    if (lid == 0)
        compressable[block_ix] = 0;
    __syncthreads();

    atomicAdd(&compressable[block_ix], inc);
    __syncthreads();
    if (lid == WGS - 1) {
        atomicAdd(cmp_size, compressable[block_ix]);
    }
}

KernelStats DGFPC::run(const FPCData &data, const FPCSettings &settings, FPCResult &result) const {
  CudaProfiling prof;
  ulong* d_values;
  unsigned* d_cmp_size;
  unsigned* d_compressable;
  size_t* d_length;
  dim3 grids (data.length/settings.wgz);
  dim3 threads (settings.wgz);
  prof.begin_mem2D();
  CHECK_CUDA(cudaMalloc((void**)&d_values, data.b_size));
  CHECK_CUDA(cudaMemcpy(d_values, data.values, data.b_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc((void**)&d_cmp_size, sizeof(unsigned)));
  CHECK_CUDA(cudaMalloc((void**)&d_length, sizeof(size_t)));
  CHECK_CUDA(cudaMemcpy(d_length, &data.length, sizeof(size_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc((void**)&d_compressable, sizeof(unsigned) * grids.x));

  prof.end_mem2D();

  prof.begin_compute();
  for(int r = 0 ; r < settings.repetitions ; r++){
    CHECK_CUDA(cudaMemset(d_cmp_size, 0, sizeof(int)));
    fpc_dg_kernel<<<grids, threads>>>(d_values, d_cmp_size,d_length,d_compressable);
    cudaError_t err = cudaGetLastError();  // check launch errors
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
    }
  }
  prof.end_compute();
  prof.begin_mem2H();
  CHECK_CUDA(cudaMemcpy(&result.size_, d_cmp_size, sizeof(unsigned), cudaMemcpyDeviceToHost));
  prof.end_mem2H();
  CHECK_CUDA(cudaFree(d_values));
  CHECK_CUDA(cudaFree(d_length));
  CHECK_CUDA(cudaFree(d_cmp_size));
  return prof.retreive(settings.repetitions);
};

REGISTER_CLASS(IFPC,DGFPC)