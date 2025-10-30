/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SOME PARTS OF THIS CODE ARE FROM NVBENCH
 */

#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP
#include "Kernel.hpp"
#include "gpu-utils.hpp"
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)

void check_cuda_error(cudaError_t error_code, const char *file, int line);

struct stream_t{
    cudaStream_t native;
};
struct event_t{
    cudaEvent_t native;
};

#endif
