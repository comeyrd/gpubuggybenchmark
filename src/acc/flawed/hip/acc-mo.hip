#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "acc-mo.hpp"
#include "hip-utils.hpp"
//Memory Overhead, Redundant Memory Operation

#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef hipcub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  input = BlockReduce(temp_storage).Sum(input);
}

__global__
void accuracy_mo_kernel(const int N, const int D, const int top_k, const float* Xdata, const int* labelData, int* accuracy){
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

KernelStats MOAccuracy::run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const{
    HipProfiling prof(settings);

    prof.begin_mem2D();
    int *d_label;
    CHECK_HIP(hipMalloc((void**)&d_label, data.label_sz_bytes));
    CHECK_HIP(hipMemcpy(d_label, data.label, data.label_sz_bytes, hipMemcpyHostToDevice));

    float *d_data;
    CHECK_HIP(hipMalloc((void**)&d_data, data.data_sz_bytes));
    CHECK_HIP(hipMemcpy(d_data, data.data, data.label_sz_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_data, data.data, data.label_sz_bytes, hipMemcpyHostToDevice));//BUG

    int *d_count;
    CHECK_HIP(hipMalloc((void**)&d_count, sizeof(int)));

    dim3 block (GPU_NUM_THREADS);

    dim3 grid (settings.grid_sz);

    
    prof.end_mem2D();
    for(int w = 0; w < settings.warmup ; w++){
      prof.begin_warmup();
      CHECK_HIP(hipMemset(d_count, 0, sizeof(int)));
      accuracy_mo_kernel<<<grid, block>>>(data.n_rows, data.ndims, data.topk, d_data, d_label, d_count);
      prof.end_warmup();
    }
    for(int r = 0 ; r < settings.repetitions ; r++){
      prof.begin_repetition();
      CHECK_HIP(hipMemset(d_count, 0, sizeof(int)));
      accuracy_mo_kernel<<<grid, block>>>(data.n_rows, data.ndims, data.topk, d_data, d_label, d_count);
      prof.end_repetition();
    }

    CHECK_HIP(hipDeviceSynchronize());
    prof.begin_mem2H();
    CHECK_HIP(hipMemcpy(&result.count, d_count, sizeof(int), hipMemcpyDeviceToHost));
    prof.end_mem2H();

    CHECK_HIP(hipFree(d_label));
    CHECK_HIP(hipFree(d_data));
    CHECK_HIP(hipFree(d_count));
    return prof.retreive();
};

REGISTER_CLASS(IAccuracy,MOAccuracy);
