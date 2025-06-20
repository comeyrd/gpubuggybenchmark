#include "fpc-reference.hpp"
#include <stdio.h>   
#include <stdlib.h> 
#include <chrono>
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

void do_fpc(int work_groupe_sz, int repeat){

  const int step = 4;
  const size_t size = (size_t)work_groupe_sz * work_groupe_sz * work_groupe_sz;
  char* cbuffer = (char*) malloc (size * step);

  srand(2);
  for (size_t i = 0; i < size*step; i++) {
    cbuffer[i] = 0xFF << (rand() % 256);
  }

  ulong *values = convertBuffer2Array (cbuffer, size, step);
  unsigned values_size = size / step;

  unsigned cmp_size = fpc_cpu(values, values_size);

  // run on the device
  unsigned cmp_size_hw; 

  bool ok = true;

  IFpc* fpc_impl = new ReferenceFpc();
  // warmup
  fpc_impl->fpc(values, &cmp_size_hw, values_size, work_groupe_sz);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc_impl->fpc(values, &cmp_size_hw, values_size, work_groupe_sz);
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
  fpc_impl->fpc2(values, &cmp_size_hw, values_size, work_groupe_sz);

  start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc_impl->fpc2(values, &cmp_size_hw, values_size, work_groupe_sz);
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

  free(values);
  free(cbuffer);
  delete fpc_impl;
}
