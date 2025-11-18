#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "acc-dr.hpp"
#include "hip-utils.hpp"
//Data Locality, using shared memory for global memory 
//TODO FIX
#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef hipcub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  input = BlockReduce(temp_storage).Sum(input);
}
__global__
void accuracy_dr_kernel(const int N, const int D, const int top_k, const float* Xdata, const int* labelData, int* accuracy){
    extern __shared__ float s_row[];  // Shared memory for one row
    int count = 0;
    
    for(int row = blockIdx.x; row < N; row += gridDim.x) {
        // Load one row into shared memory
        for (int col = threadIdx.x; col < D; col += blockDim.x) {
            s_row[col] = Xdata[row * D + col];
        }
        __syncthreads();
        
        const int label = labelData[row];
        const float label_pred = s_row[label];
        int ngt = 0;
        
        for (int col = threadIdx.x; col < D; col += blockDim.x) {
            const float pred = s_row[col];
            if (pred > label_pred || (pred == label_pred && col <= label)) {
                ++ngt;
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
void DRAccuracy::setup(){
  CHECK_HIP(hipMalloc((void**)&d_label, m_data->label_sz_bytes));
  CHECK_HIP(hipMemcpy(d_label, m_data->label, m_data->label_sz_bytes, hipMemcpyHostToDevice));

  CHECK_HIP(hipMalloc((void**)&d_data, m_data->data_sz_bytes));
  CHECK_HIP(hipMemcpy(d_data, m_data->data, m_data->data_sz_bytes, hipMemcpyHostToDevice));

  CHECK_HIP(hipMalloc((void**)&d_count, sizeof(int)));
  block = dim3(GPU_NUM_THREADS);
  int grid_sz = (m_data->n_rows + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;
  grid = dim3(grid_sz);
}

void DRAccuracy::reset(){
  CHECK_HIP(hipMemset(d_count, 0, sizeof(int)));
}
void DRAccuracy::run(stream_t* s){
  accuracy_dr_kernel<<<grid, block,m_data->ndims*sizeof(float), s->native>>>(m_data->n_rows, m_data->ndims, m_data->topk, d_data, d_label, d_count);
}

void DRAccuracy::teardown(AccuracyResult &_result){
  CHECK_HIP(hipMemcpy(&_result.count, d_count, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d_label));
  CHECK_HIP(hipFree(d_data));
  CHECK_HIP(hipFree(d_count));
}

REGISTER_CLASS(IAccuracy,DRAccuracy);
