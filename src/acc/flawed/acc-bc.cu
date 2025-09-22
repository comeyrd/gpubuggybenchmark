#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>
#include "acc-bc.hpp"
#include "cuda-utils.hpp"
//Not Enqueing, Using blocking calls for memory copy or kernel executions

#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef cub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
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

KernelStats BCAccuracy::run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const{
    CudaProfiling prof;

    prof.begin_mem2D();
    int *d_label;
    CHECK_CUDA(cudaMalloc((void**)&d_label, data.label_sz_bytes));
    CHECK_CUDA(cudaDeviceSynchronize());//BUG
    CHECK_CUDA(cudaMemcpy(d_label, data.label, data.label_sz_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());//BUG
    float *d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, data.data_sz_bytes));
    CHECK_CUDA(cudaDeviceSynchronize());//BUG
    CHECK_CUDA(cudaMemcpy(d_data, data.data, data.data_sz_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());//BUG

    int *d_count;
    CHECK_CUDA(cudaMalloc((void**)&d_count, sizeof(int)));
    CHECK_CUDA(cudaDeviceSynchronize());//BUG

    dim3 block (GPU_NUM_THREADS);

    dim3 grid (settings.grid_sz);

    
    CHECK_CUDA(cudaDeviceSynchronize());//BUG
    prof.end_mem2D();
    prof.begin_compute();
    for(int r = 0 ; r < settings.repetitions ; r++){
      CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));
      accuracy_bc_kernel<<<grid, block>>>(data.n_rows, data.ndims, data.topk, d_data, d_label, d_count);
    }
    CHECK_CUDA(cudaDeviceSynchronize());//BUG
    prof.end_compute();

    CHECK_CUDA(cudaDeviceSynchronize());
    prof.begin_mem2H();
    CHECK_CUDA(cudaMemcpy(&result.count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());//BUG
    prof.end_mem2H();

    CHECK_CUDA(cudaFree(d_label));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_count));
    return prof.retreive(settings.repetitions);
};

REGISTER_CLASS(IAccuracy,BCAccuracy);