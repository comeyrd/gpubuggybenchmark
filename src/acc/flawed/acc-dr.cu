#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>
#include "acc-dr.hpp"
#include "cuda-utils.hpp"
//Data Locality, using register memory for shared or global memory (register spilling)
//TODO FIX
#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef cub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  input = BlockReduce(temp_storage).Sum(input);
}

__global__
void accuracy_dr_kernel(const int N, const int D, const int top_k, const float* Xdata, const int* labelData, int* accuracy){
  int count = 0;
  float l_xdata[10000] ;
  for (int i = threadIdx.x; i < D*N; i += blockDim.x) {
    l_xdata[i] = Xdata[i];
  }

  __syncthreads();
  for(int row = blockIdx.x; row < N; row += gridDim.x) {
    const int label = labelData[row];
    const float label_pred = l_xdata[row * D + label];
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = l_xdata[row * D + col];
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

KernelStats DRAccuracy::accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const{
    CudaProfiling prof;

    prof.begin_mem2D();
    int *d_label;
    CHECK_CUDA(cudaMalloc((void**)&d_label, aData.label_sz_bytes));
    CHECK_CUDA(cudaMemcpy(d_label, aData.label, aData.label_sz_bytes, cudaMemcpyHostToDevice));
    
    float *d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, aData.label_sz_bytes));
    CHECK_CUDA(cudaMemcpy(d_data, aData.data, aData.label_sz_bytes, cudaMemcpyHostToDevice));

    int *d_count;
    CHECK_CUDA(cudaMalloc((void**)&d_count, sizeof(int)));

    dim3 block (GPU_NUM_THREADS);

    dim3 grid (aSettings.grid_sz);

    CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));
    prof.end_mem2D();
    prof.begin_compute();
    accuracy_dr_kernel<<<grid, block>>>(aData.n_rows, aData.ndims, aData.topk, d_data, d_label, d_count);
    prof.end_compute();

    CHECK_CUDA(cudaDeviceSynchronize());
    prof.begin_mem2H();
    CHECK_CUDA(cudaMemcpy(&aResult.count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    prof.end_mem2H();

    CHECK_CUDA(cudaFree(d_label));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_count));
    return prof.retreive();
};

//REGISTER_CLASS(IAccuracy,DRAccuracy);