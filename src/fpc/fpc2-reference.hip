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

__device__
unsigned f1(ulong value, bool* mask) {
  if (value == 0) {
    *mask = 1;
  } 
  return 1;
}

__device__
unsigned f2(ulong value, bool* mask) {
  if (my_abs((int)(value)) <= 0xFF) *mask = 1;
  return 1;
}

__device__
unsigned f3(ulong value, bool* mask) {
  if (my_abs((int)(value)) <= 0xFFFF) *mask = 1;
  return 2;
}

__device__
unsigned f4(ulong value, bool* mask) {
  if (((value) & 0xFFFF) == 0 ) *mask = 1;
  return 2;
}

__device__
unsigned f5(ulong value, bool* mask) {
  if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF && 
      my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) 
    *mask = 1;
  return 2;
}

__device__
unsigned f6(ulong value, bool* mask) {
  unsigned byte0 = (value) & 0xFF;
  unsigned byte1 = (value >> 8) & 0xFF;
  unsigned byte2 = (value >> 16) & 0xFF;
  unsigned byte3 = (value >> 24) & 0xFF;
  if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) 
    *mask = 1;
  return 1;
}

__device__
unsigned f7(ulong value, bool* mask) {
  *mask = 1;
  return 4;
}

}

__global__ void
fpc2_reference_kernel(const ulong *values, unsigned *cmp_size, size_t* length) {

  __shared__ unsigned compressable;
  int lid = threadIdx.x;
  int WGS = blockDim.x;
  int gid = blockIdx.x*WGS+lid;
  if(gid > *length) return;
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


  if (lid == 0) compressable = 0;
  __syncthreads();

  atomicAdd(&compressable, inc);
  __syncthreads();
  if (lid == WGS-1) {
    atomicAdd(cmp_size, compressable);
  }
}

void ReferenceFPC2::setup() {
    CHECK_HIP(hipMalloc((void **)&d_values, m_data->b_size));
    CHECK_HIP(hipMemcpy(d_values, m_data->values, m_data->b_size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMalloc((void **)&d_cmp_size, sizeof(unsigned)));
    CHECK_HIP(hipMalloc((void **)&d_length, sizeof(size_t)));
    CHECK_HIP(hipMemcpy(d_length, &m_data->length, sizeof(size_t), hipMemcpyHostToDevice));
    grids = dim3(m_data->length / m_data->wgz);
    threads = dim3(m_data->wgz);
}

void ReferenceFPC2::reset() {
    CHECK_HIP(hipMemset(d_cmp_size, 0, sizeof(int)));
}

void ReferenceFPC2::run(stream_t *s) {
    fpc2_reference_kernel<<<grids, threads,0,s->native>>>(d_values, d_cmp_size, d_length);
}
void ReferenceFPC2::teardown(FPCResult &_result){
    CHECK_HIP(hipMemcpy(&_result.size_, d_cmp_size, sizeof(unsigned), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(d_values));
    CHECK_HIP(hipFree(d_length));
    CHECK_HIP(hipFree(d_cmp_size));
}

REGISTER_CLASS(IFPC, ReferenceFPC2)
