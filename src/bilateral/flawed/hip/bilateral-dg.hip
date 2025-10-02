#include "bilateral-dg.hpp"
#include "gpu-utils.hpp"
#include "hip-utils.hpp"
//Too much global memory access
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

      // range kernel for smoothing differences in intensities
      float range = -(in[id]-in[id_w]) * (in[id]-in[id_w]) / (2.f * variance_I);//BUG

      // spatial (or domain) kernel for smoothing differences in coordinates
      float spatial = -((idk-idx)*(idk-idx) + (idl-idy)*(idl-idy)) /
                      (2.f * variance_spatial);

      // the weight is assigned using the spatial closeness (using the spatial kernel) 
      // and the intensity difference (using the range kernel)
      float weight = a_square * expf(spatial + range);

      normalization += weight;
      res += (in[id_w] * weight);
    }
  }
  out[id] = res/normalization;
}
KernelStats DGBilateral::run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const {

  float *d_src, *d_dst;
  HipProfiling prof(settings);
  prof.begin_mem2D();
  CHECK_HIP(hipMalloc((void**)&d_dst, data.size * sizeof(float)));
  CHECK_HIP(hipMalloc((void**)&d_src, data.size * sizeof(float)));

  CHECK_HIP(hipMemcpy(d_src, data.image, data.size * sizeof(float), hipMemcpyHostToDevice)); 
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
  CHECK_HIP(hipMemcpy(result.image, d_dst, data.size * sizeof(float), hipMemcpyDeviceToHost)); 
  prof.end_mem2H();
  return prof.retreive();
};
REGISTER_CLASS(IBilateral,DGBilateral);
