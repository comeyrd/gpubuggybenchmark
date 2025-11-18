#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "acc-mc.hpp"
#include "hip-utils.hpp"
//Memory Overhead, Too much data movement between the host and the device

#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef hipcub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  input = BlockReduce(temp_storage).Sum(input);
}

__global__
void accuracy_mc_kernel(const int N, const int D, const int top_k, const float* Xdata, const int* labelData, int* accuracy){
  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    BlockReduce(ngt);
    if (ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { 
    atomicAdd(accuracy, count);
  }
}
void MCAccuracy::setup(){
  CHECK_HIP(hipMalloc((void**)&d_label, m_data->label_sz_bytes));
  CHECK_HIP(hipMemcpy(d_label, m_data->label, m_data->label_sz_bytes, hipMemcpyHostToDevice));

  CHECK_HIP(hipMalloc((void**)&d_data, m_data->data_sz_bytes));
  CHECK_HIP(hipMemcpy(d_data, m_data->data, m_data->data_sz_bytes, hipMemcpyHostToDevice));

  CHECK_HIP(hipMalloc((void**)&d_count, sizeof(int)));
  block = dim3(GPU_NUM_THREADS);
  int grid_sz = (m_data->n_rows + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;
  grid = dim3(grid_sz);
}

void MCAccuracy::reset(){
  CHECK_HIP(hipMemset(d_count, 0, sizeof(int)));
}
void MCAccuracy::run(stream_t* s){
  accuracy_mc_kernel<<<grid, block,0, s->native>>>(m_data->n_rows, m_data->ndims, m_data->topk, d_data, d_label, d_count);
}

void MCAccuracy::teardown(AccuracyResult &_result){
  CHECK_HIP(hipMemcpy(&_result.count, d_count, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d_label));
  CHECK_HIP(hipFree(d_data));
  CHECK_HIP(hipFree(d_count));
}

REGISTER_CLASS(IAccuracy,MCAccuracy);
