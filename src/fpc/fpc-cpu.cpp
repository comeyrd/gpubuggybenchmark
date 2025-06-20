#include <stdio.h>      /* defines printf for tests */
#include <stdlib.h> 
#include <chrono>

typedef unsigned long ulong;

unsigned my_abs_cpu( int x )
{
  unsigned t = x >> 31;
  return (x ^ t) - t;
}

unsigned fpc_cpu(ulong *values, unsigned size )
{
  unsigned compressable = 0;
  unsigned i;
  for (i = 0; i < size; i++) {
    // 000
    if(values[i] == 0){
      compressable += 1;
      continue;
    }
    // 001 010
    if(my_abs_cpu((int)(values[i])) <= 0xFF){
      compressable += 1;
      continue;
    }
    // 011
    if(my_abs_cpu((int)(values[i])) <= 0xFFFF){
      compressable += 2;
      continue;
    }
    //100  
    if(((values[i]) & 0xFFFF) == 0 ){
      compressable += 2;
      continue;
    }
    //101
    if( my_abs_cpu((int)((values[i]) & 0xFFFF)) <= 0xFF
        && my_abs_cpu((int)((values[i] >> 16) & 0xFFFF)) <= 0xFF){
      compressable += 2;
      continue;
    }
    //110
    unsigned byte0 = (values[i]) & 0xFF;
    unsigned byte1 = (values[i] >> 8) & 0xFF;
    unsigned byte2 = (values[i] >> 16) & 0xFF;
    unsigned byte3 = (values[i] >> 24) & 0xFF;
    if(byte0 == byte1 && byte0 == byte2 && byte0 == byte3){
      compressable += 1;
      continue;
    }
    //111
    compressable += 4;
  }
  return compressable;
}