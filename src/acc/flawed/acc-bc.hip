#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "acc-bc.hpp"
#include "hip-utils.hpp"
//Not Enqueing, Using blocking calls for memory copy or kernel executions

#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef hipcub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  input = BlockReduce(temp_storage).Sum(input);
}

__global__
void accuracy_bc_kernel(const int N, const int D, const int top_k, const float* Xdata, const int* labelData, int* accuracy){
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

KernelStats BCAccuracy::accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const{
    HipProfiling prof;

    prof.begin_mem2D();
    int *d_label;
    CHECK_HIP(hipMalloc((void**)&d_label, aData.label_sz_bytes));
    CHECK_HIP(hipDeviceSynchronize());//BUG
    CHECK_HIP(hipMemcpy(d_label, aData.label, aData.label_sz_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipDeviceSynchronize());//BUG
    float *d_data;
    CHECK_HIP(hipMalloc((void**)&d_data, aData.label_sz_bytes));
    CHECK_HIP(hipDeviceSynchronize());//BUG
    CHECK_HIP(hipMemcpy(d_data, aData.data, aData.label_sz_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipDeviceSynchronize());//BUG

    int *d_count;
    CHECK_HIP(hipMalloc((void**)&d_count, sizeof(int)));
    CHECK_HIP(hipDeviceSynchronize());//BUG

    dim3 block (GPU_NUM_THREADS);

    dim3 grid (aSettings.grid_sz);

    CHECK_HIP(hipMemset(d_count, 0, sizeof(int)));
    CHECK_HIP(hipDeviceSynchronize());//BUG
    prof.end_mem2D();
    prof.begin_compute();
    accuracy_bc_kernel<<<grid, block>>>(aData.n_rows, aData.ndims, aData.topk, d_data, d_label, d_count);
    CHECK_HIP(hipDeviceSynchronize());//BUG
    prof.end_compute();

    CHECK_HIP(hipDeviceSynchronize());
    prof.begin_mem2H();
    CHECK_HIP(hipMemcpy(&aResult.count, d_count, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipDeviceSynchronize());//BUG
    prof.end_mem2H();

    CHECK_HIP(hipFree(d_label));
    CHECK_HIP(hipFree(d_data));
    CHECK_HIP(hipFree(d_count));
    return prof.retreive();
};

REGISTER_CLASS(IAccuracy,BCAccuracy);
