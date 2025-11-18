#include "cuda-utils.hpp"
#include "fpc-reference.hpp"
#include <chrono>
#include <cuda.h>
#include <stdio.h> /* defines printf for tests */
#include <stdlib.h>

namespace {
    __device__ unsigned my_abs(int x) {
        unsigned t = x >> 31;
        return (x ^ t) - t;
    }
}

__global__ void
fpc_reference_kernel(const ulong *values, unsigned *cmp_size, size_t *length) {
    __shared__ unsigned compressable;
    int lid = threadIdx.x;
    int WGS = blockDim.x;
    size_t gid = blockIdx.x * WGS + lid;
    if (gid > *length)
        return;
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

void ReferenceFPC::setup() {
    CHECK_CUDA(cudaMalloc((void **)&d_values, m_data->b_size));
    CHECK_CUDA(cudaMemcpy(d_values, m_data->values, m_data->b_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void **)&d_cmp_size, sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc((void **)&d_length, sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_length, &m_data->length, sizeof(size_t), cudaMemcpyHostToDevice));
    grids = dim3(m_data->length / m_data->wgz);
    threads = dim3(m_data->wgz);
}

void ReferenceFPC::reset() {
    CHECK_CUDA(cudaMemset(d_cmp_size, 0, sizeof(int)));
}

void ReferenceFPC::run(stream_t *s) {
    fpc_reference_kernel<<<grids, threads,0,s->native>>>(d_values, d_cmp_size, d_length);
}
void ReferenceFPC::teardown(FPCResult &_result){
    CHECK_CUDA(cudaMemcpy(&_result.size_, d_cmp_size, sizeof(unsigned), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_length));
    CHECK_CUDA(cudaFree(d_cmp_size));
}

REGISTER_CLASS(IFPC, ReferenceFPC)