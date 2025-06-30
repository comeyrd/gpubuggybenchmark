#include "fpc-reference.hpp"
#include <stdio.h>   
#include <stdlib.h> 
#include <chrono>
#include <memory>
#include <iostream>

#include "fpc.hpp"
ulong* convertBuffer2Array (char *cbuffer, unsigned size, unsigned step)
{
  unsigned i,j; 
  ulong * values = NULL;
  posix_memalign((void**)&values, 1024, sizeof(ulong)*size/step);
  for (i = 0; i < size / step; i++) {
    values[i] = 0;    // Initialize all elements to zero.
  }
  for (i = 0; i < size; i += step ){
    for (j = 0; j < step; j++){
      values[i / step] += (ulong)((unsigned char)cbuffer[i + j]) << (8*j);
    }
  }
  return values;
}

void do_fpc(int work_group_sz, int repeat){

  const int step = 4;
  const size_t size = (size_t)work_group_sz * work_group_sz * work_group_sz;
  char* cbuffer = (char*) malloc (size * step);

  srand(2);
  for (size_t i = 0; i < size*step; i++) {
    cbuffer[i] = 0xFF << (rand() % 256);
  }

  ulong *values = convertBuffer2Array (cbuffer, size, step);
  unsigned values_size = size / step;

  unsigned cmp_size = fpc_cpu(values, values_size);
  kernel_umap<IFpc> kernels = Manager<IFpc>::instance()->getKernels();

  for (const auto &[name, k_func] : kernels) {
    std::cout <<" Doing Kernel "<< name << std::endl;
    run_fpc_impl(k_func,values,values_size,cmp_size, work_group_sz, repeat);
  }


  free(values);
  free(cbuffer);
}


void run_fpc_impl(std::shared_ptr<IFpc> fpc_impl, ulong* values, unsigned values_size, int cmp_size, int work_group_sz, int repeat){

// run on the device
  unsigned cmp_size_hw; 

  bool ok = true;
  // warmup
  fpc_impl->fpc(values, &cmp_size_hw, values_size, work_group_sz);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc_impl->fpc(values, &cmp_size_hw, values_size, work_group_sz);
    if (cmp_size_hw != cmp_size) {
      printf("fpc failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  // warmup
  fpc_impl->fpc2(values, &cmp_size_hw, values_size, work_group_sz);

  start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc_impl->fpc2(values, &cmp_size_hw, values_size, work_group_sz);
    if (cmp_size_hw != cmp_size) {
      printf("fpc2 failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc2: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  printf("%s\n", ok ? "PASS" : "FAIL");

}
REGISTER_CLASS(IKernel,IFpc);