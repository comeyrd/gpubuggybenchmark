#ifndef CUDA_UTILS_HPP  
#define  CUDA_UTILS_HPP
#include "Kernel.hpp"

#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)

void check_cuda_error(cudaError_t error_code,const char* file, int line);

class CudaProfiling{
    private:
    cudaEvent_t memstart2D;
    cudaEvent_t memstop2D;
    cudaEvent_t memstart2H;
    cudaEvent_t memstop2H;
    cudaEvent_t computestart;
    cudaEvent_t computestop;
    bool destroy = false;
    public:
    CudaProfiling();
    ~CudaProfiling();
    void begin_mem2D();
    void end_mem2D();
    void begin_mem2H();
    void end_mem2H();
    void begin_compute();
    void end_compute();
    KernelStats retreive(); 
};

//TODO Update the tools to use "nsys" and "ncu" and other.

#endif

