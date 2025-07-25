#include "fpc-rc.hpp"
#include <chrono>
#include <cuda.h>
#include <stdio.h> /* defines printf for tests */
#include <stdlib.h>
namespace {
__device__ unsigned my_abs(int x) {
    unsigned t = x >> 31;
    return (x ^ t) - t;
}

__device__ unsigned f1(ulong value, bool *mask) {
    if (value == 0) {
        *mask = 1;
    }
    return 1;
}

__device__ unsigned f2(ulong value, bool *mask) {
    if (my_abs((int)(value)) <= 0xFF)
        *mask = 1;
    return 1;
}

__device__ unsigned f3(ulong value, bool *mask) {
    if (my_abs((int)(value)) <= 0xFFFF)
        *mask = 1;
    return 2;
}

__device__ unsigned f4(ulong value, bool *mask) {
    if (((value) & 0xFFFF) == 0)
        *mask = 1;
    return 2;
}

__device__ unsigned f5(ulong value, bool *mask) {
    if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF &&
        my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF)
        *mask = 1;
    return 2;
}

__device__ unsigned f6(ulong value, bool *mask) {
    unsigned byte0 = (value) & 0xFF;
    unsigned byte1 = (value >> 8) & 0xFF;
    unsigned byte2 = (value >> 16) & 0xFF;
    unsigned byte3 = (value >> 24) & 0xFF;
    if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3)
        *mask = 1;
    return 1;
}

__device__ unsigned f7(ulong value, bool *mask) {
    *mask = 1;
    return 4;
}
}
__global__ void
fpc_rc_kernel(const ulong *values, unsigned *cmp_size) {
    __shared__ unsigned compressable;
    int lid = threadIdx.x;
    int WGS = blockDim.x;
    int gid = blockIdx.x * WGS + lid;

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
    //__syncthreads();BUG

    atomicAdd(&compressable, inc);
    //__syncthreads();BUG
    if (lid == WGS - 1) {
        atomicAdd(cmp_size, compressable);
    }
}

__global__ void
fpc2_rc_kernel(const ulong *values, unsigned *cmp_size) {
    __shared__ unsigned compressable;
    int lid = threadIdx.x;
    int WGS = blockDim.x;
    int gid = blockIdx.x * WGS + lid;

    unsigned inc;

    bool m1 = 0;
    bool m2 = 0;
    bool m3 = 0;
    bool m4 = 0;
    bool m5 = 0;
    bool m6 = 0;
    bool m7 = 0;

    ulong value = values[gid];
    unsigned inc1 = f1(value, &m1);
    unsigned inc2 = f2(value, &m2);
    unsigned inc3 = f3(value, &m3);
    unsigned inc4 = f4(value, &m4);
    unsigned inc5 = f5(value, &m5);
    unsigned inc6 = f6(value, &m6);
    unsigned inc7 = f7(value, &m7);

    if (m1)
        inc = inc1;
    else if (m2)
        inc = inc2;
    else if (m3)
        inc = inc3;
    else if (m4)
        inc = inc4;
    else if (m5)
        inc = inc5;
    else if (m6)
        inc = inc6;
    else
        inc = inc7;

    if (lid == 0)
        compressable = 0;
    //__syncthreads();BUG
    atomicAdd(&compressable, inc);
    //__syncthreads();BUG
    if (lid == WGS - 1) {
        atomicAdd(cmp_size, compressable);
    }
}

void fpc_rc(const ulong *values, unsigned *cmp_size_hw, const int values_size, const int wgs) {
    *cmp_size_hw = 0;
    ulong *d_values;
    unsigned *d_cmp_size;
    cudaMalloc((void **)&d_values, values_size * sizeof(ulong));
    cudaMemcpy(d_values, values, values_size * sizeof(ulong), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_cmp_size, sizeof(unsigned));
    cudaMemcpy(d_cmp_size, cmp_size_hw, sizeof(unsigned), cudaMemcpyHostToDevice);

    dim3 grids(values_size / wgs);
    dim3 threads(wgs);

    fpc_rc_kernel<<<grids, threads>>>(d_values, d_cmp_size);

    cudaMemcpy(cmp_size_hw, d_cmp_size, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaFree(d_values);
    cudaFree(d_cmp_size);
}

void fpc2_rc(const ulong *values, unsigned *cmp_size_hw, const int values_size, const int wgs) {
    *cmp_size_hw = 0;
    ulong *d_values;
    unsigned *d_cmp_size;
    cudaMalloc((void **)&d_values, values_size * sizeof(ulong));
    cudaMemcpy(d_values, values, values_size * sizeof(ulong), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_cmp_size, sizeof(unsigned));
    cudaMemcpy(d_cmp_size, cmp_size_hw, sizeof(unsigned), cudaMemcpyHostToDevice);

    dim3 grids(values_size / wgs);
    dim3 threads(wgs);

    fpc2_rc_kernel<<<grids, threads>>>(d_values, d_cmp_size);

    cudaMemcpy(cmp_size_hw, d_cmp_size, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaFree(d_values);
    cudaFree(d_cmp_size);
}
REGISTER_CLASS(IFpc,RCFpc)
