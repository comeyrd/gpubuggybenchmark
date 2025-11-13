#include "acc-cl.hpp"
#include "hip-utils.hpp"
#include <chrono>
#include <hipcub/hipcub.hpp>
#include <hip/hip_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
// Inefficient Cache access, non linear data access
// TODO FIX - IS buggy
#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
    typedef hipcub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    input = BlockReduce(temp_storage).Sum(input);
}
__global__
void accuracy_cl_kernel(const int N, const int D, const int top_k, const float* Xdata, const int* labelData, int* accuracy){
    int count = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        const int label = labelData[row];
        const float label_pred = Xdata[row * D + label];
        int ngt = 0;
        //BUG: serialize access - only one thread works at a time
        for (int col = 0; col < D; col++) {
            if (col % blockDim.x == threadIdx.x) {
                const float pred = Xdata[row * D + col];
                if (pred > label_pred || (pred == label_pred && col <= label)) {
                    ++ngt;
                }
            }
        }
        BlockReduce(ngt);
        if (threadIdx.x == 0 && ngt <= top_k) {
            ++count;
        }
        __syncthreads();
    }
    
    BlockReduce(count);
    
    if (threadIdx.x == 0) { 
        atomicAdd(accuracy, count);
    }
}
void CLAccuracy::setup() {
    CHECK_HIP(hipMalloc((void **)&d_label, m_data->label_sz_bytes));
    CHECK_HIP(hipMemcpy(d_label, m_data->label, m_data->label_sz_bytes, hipMemcpyHostToDevice));

    CHECK_HIP(hipMalloc((void **)&d_data, m_data->data_sz_bytes));
    CHECK_HIP(hipMemcpy(d_data, m_data->data, m_data->data_sz_bytes, hipMemcpyHostToDevice));

    CHECK_HIP(hipMalloc((void **)&d_count, sizeof(int)));
    block = dim3(GPU_NUM_THREADS);
    int grid_sz = (m_data->n_rows + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;
    grid = dim3(grid_sz);
}

void CLAccuracy::reset() {
    CHECK_HIP(hipMemset(d_count, 0, sizeof(int)));
}
void CLAccuracy::run(stream_t *s) {
    accuracy_cl_kernel<<<grid, block, 0, s->native>>>(m_data->n_rows, m_data->ndims, m_data->topk, d_data, d_label, d_count);
}

void CLAccuracy::teardown(AccuracyResult &_result) {
    CHECK_HIP(hipMemcpy(&_result.count, d_count, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(d_label));
    CHECK_HIP(hipFree(d_data));
    CHECK_HIP(hipFree(d_count));
}

REGISTER_CLASS(IAccuracy, CLAccuracy);
