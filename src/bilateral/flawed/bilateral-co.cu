#include "bilateral-co.hpp"
#include "gpu-utils.hpp"
#include "cuda-utils.hpp"
//Costly way of doing a computation
template<int R>
__global__ void bilateralFilter(
    const float *__restrict__ in,
    float *__restrict__ out,
    int w, 
    int h, 
    float a_square,
    float variance_I,
    float variance_spatial)
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  const int idy = blockIdx.y*blockDim.y + threadIdx.y;

  if(idx >= w || idy >= h) return;

  int id = idy*w + idx;
  float I = in[id];
  float res = 0;
  float normalization = 0;

  // window centered at the coordinate (idx, idy)
  #pragma unroll
  for(int i = -R; i <= R; i++) {
    #pragma unroll
    for(int j = -R; j <= R; j++) {

      int idk = idx+i;
      int idl = idy+j;

      // mirror edges
      if( idk < 0) idk = -idk;
      if( idl < 0) idl = -idl;
      if( idk > w - 1) idk = w - 1 - i;
      if( idl > h - 1) idl = h - 1 - j;

      int id_w = idl*w + idk;
      float I_w = in[id_w];

      // range kernel for smoothing differences in intensities
      float range = -(I-I_w) * (I-I_w) / (2.f * variance_I);

      // spatial (or domain) kernel for smoothing differences in coordinates
      float spatial = -((idk-idx)*(idk-idx) + (idl-idy)*(idl-idy)) /
                      (2.f * variance_spatial);

      // the weight is assigned using the spatial closeness (using the spatial kernel) 
      // and the intensity difference (using the range kernel)
      float weight = a_square * expf(spatial) + expf(range);//BUG

      normalization += weight;
      res += (I_w * weight);
    }
  }
  out[id] = res/normalization;
}
KernelStats COBilateral::run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const {

  float *d_src, *d_dst;
  CudaProfiling prof(settings);
  prof.begin_mem2D();
  CHECK_CUDA(cudaMalloc((void**)&d_dst, data.size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_src, data.size * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_src, data.image, data.size * sizeof(float), cudaMemcpyHostToDevice)); 
  prof.end_mem2D();
  dim3 threads (16, 16);
  dim3 blocks ((data.width+15)/16, (data.height+15)/16);
  for(int w = 0; w < settings.warmup ; w++){
    prof.begin_warmup();
    bilateralFilter<4><<<blocks, threads>>>(d_src, d_dst, data.width, data.height, settings.a_square, settings.variance_I, settings.variance_spatial);
    prof.end_warmup();
  }
  for(int r = 0 ; r < settings.repetitions ; r++){
    prof.begin_repetition();
    bilateralFilter<4><<<blocks, threads>>>(d_src, d_dst, data.width, data.height, settings.a_square, settings.variance_I, settings.variance_spatial);
    prof.end_repetition();
  }
  prof.begin_mem2H();
  CHECK_CUDA(cudaMemcpy(result.image, d_dst, data.size * sizeof(float), cudaMemcpyDeviceToHost)); 
  prof.end_mem2H();
  return prof.retreive();
};
REGISTER_CLASS(IBilateral,COBilateral);