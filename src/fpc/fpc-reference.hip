#include <stdio.h>      /* defines printf for tests */
#include <stdlib.h> 
#include <chrono>
#include <hip/hip_runtime.h>
#include "fpc-reference.hpp"
#include "hip-utils.hpp"

namespace {
__device__
unsigned my_abs ( int x )
{
  unsigned t = x >> 31;
  return (x ^ t) - t;
}
}

__global__ void
fpc_reference_kernel(const ulong *values, unsigned *cmp_size, size_t* length) {
    __shared__ unsigned compressable;
    int lid = threadIdx.x;
    int WGS = blockDim.x;
    size_t gid = blockIdx.x * WGS + lid;
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
        compressable = 0;
    __syncthreads();

    atomicAdd(&compressable, inc);
    __syncthreads();
    if (lid == WGS - 1) {
        atomicAdd(cmp_size, compressable);
    }
}

KernelStats ReferenceFPC::run(const FPCData &data, const FPCSettings &settings, FPCResult &result) const {
  HipProfiling prof(settings);
  ulong* d_values;
  unsigned* d_cmp_size;
  size_t* d_length;
  prof.begin_mem2D();
  CHECK_HIP(hipMalloc((void**)&d_values, data.b_size));
  CHECK_HIP(hipMemcpy(d_values, data.values, data.b_size, hipMemcpyHostToDevice));
  CHECK_HIP(hipMalloc((void**)&d_cmp_size, sizeof(unsigned)));
  CHECK_HIP(hipMalloc((void**)&d_length, sizeof(size_t)));
  CHECK_HIP(hipMemcpy(d_length, &data.length, sizeof(size_t), hipMemcpyHostToDevice));

  prof.end_mem2D();
  dim3 grids (data.length/settings.wgz);
  dim3 threads (settings.wgz);
  for(int w = 0; w < settings.warmup ; w++){
    prof.begin_warmup();
    CHECK_HIP(hipMemset(d_cmp_size, 0, sizeof(int)));
    fpc_reference_kernel<<<grids, threads>>>(d_values, d_cmp_size,d_length);
    prof.end_warmup();
  }
  for(int r = 0 ; r < settings.repetitions ; r++){
    prof.begin_repetition();
    CHECK_HIP(hipMemset(d_cmp_size, 0, sizeof(int)));
    fpc_reference_kernel<<<grids, threads>>>(d_values, d_cmp_size,d_length);
    prof.end_repetition();
  }
  prof.begin_mem2H();
  CHECK_HIP(hipMemcpy(&result.size_, d_cmp_size, sizeof(unsigned), hipMemcpyDeviceToHost));
  prof.end_mem2H();
  CHECK_HIP(hipFree(d_values));
  CHECK_HIP(hipFree(d_length));
  CHECK_HIP(hipFree(d_cmp_size));
  return prof.retreive();
};

REGISTER_CLASS(IFPC,ReferenceFPC)
