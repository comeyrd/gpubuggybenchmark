#include "bilateral-cl.hpp"
#include "gpu-utils.hpp"
#include "hip-utils.hpp"
//NON linear data access
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

      int id_w = idl*h + idk;//BUG
      float I_w = in[id_w];

      // range kernel for smoothing differences in intensities
      float range = -(I-I_w) * (I-I_w) / (2.f * variance_I);

      // spatial (or domain) kernel for smoothing differences in coordinates
      float spatial = -((idk-idx)*(idk-idx) + (idl-idy)*(idl-idy)) /
                      (2.f * variance_spatial);

      // the weight is assigned using the spatial closeness (using the spatial kernel) 
      // and the intensity difference (using the range kernel)
      float weight = a_square * expf(spatial + range);

      normalization += weight;
      res += (I_w * weight);
    }
  }
  out[id] = res/normalization;
}
void CLBilateral::setup() {
  CHECK_HIP(hipMalloc((void **)&d_dst, m_data->b_size));
  CHECK_HIP(hipMalloc((void **)&d_src, m_data->b_size));
  CHECK_HIP(hipMemcpy(d_src, m_data->image, m_data->b_size, hipMemcpyHostToDevice));
  threads = dim3(16, 16);
  blocks = dim3((m_data->width + 15) / 16, (m_data->height + 15) / 16);
}

void CLBilateral::run(stream_t* s) {
    bilateralFilter<4><<<blocks, threads, 0, s->native>>>(d_src, d_dst, m_data->width, m_data->height, m_data->a_square, m_data->variance_I, m_data->variance_spatial);
};

void CLBilateral::teardown(BilateralData &_result) {
  CHECK_HIP(hipMemcpy(_result.image, d_dst, m_data->b_size, hipMemcpyDeviceToHost)); 
  hipFree(d_dst);
  hipFree(d_src);
};

REGISTER_CLASS(IBilateral,CLBilateral);
