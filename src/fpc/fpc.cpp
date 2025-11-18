#include "fpc.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <random>

void FPCData::generate_random() {
    std::default_random_engine g(123);
    std::uniform_int_distribution<unsigned long> data_dist(0, std::numeric_limits<unsigned long>::max());
    for (size_t i = 0; i < length; i++) {
        values[i] = data_dist(g);
    }
}

unsigned my_abs_cpu( int x )
{
  unsigned t = x >> 31;
  return (x ^ t) - t;
}

void FPC::run_cpu() {
    unsigned compressable = 0;
    unsigned i;
    for (i = 0; i < m_data.length; i++) {
        // 000
        if (m_data.values[i] == 0) {
            compressable += 1;
            continue;
        }
        // 001 010
        if (my_abs_cpu((int)(m_data.values[i])) <= 0xFF) {
            compressable += 1;
            continue;
        }
        // 011
        if (my_abs_cpu((int)(m_data.values[i])) <= 0xFFFF) {
            compressable += 2;
            continue;
        }
        // 100
        if (((m_data.values[i]) & 0xFFFF) == 0) {
            compressable += 2;
            continue;
        }
        // 101
        if (my_abs_cpu((int)((m_data.values[i]) & 0xFFFF)) <= 0xFF && my_abs_cpu((int)((m_data.values[i] >> 16) & 0xFFFF)) <= 0xFF) {
            compressable += 2;
            continue;
        }
        // 110
        unsigned byte0 = (m_data.values[i]) & 0xFF;
        unsigned byte1 = (m_data.values[i] >> 8) & 0xFF;
        unsigned byte2 = (m_data.values[i] >> 16) & 0xFF;
        unsigned byte3 = (m_data.values[i] >> 24) & 0xFF;
        if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) {
            compressable += 1;
            continue;
        }
        // 111
        compressable += 4;
    }
    m_cpu_result.size_ = compressable;
}
REGISTER_CLASS(I_IKernel, FPC)