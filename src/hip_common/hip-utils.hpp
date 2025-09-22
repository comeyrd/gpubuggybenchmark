#ifndef HIP_UTILS_HPP  
#define  HIP_UTILS_HPP
#include "Kernel.hpp"
#include <hip/hip_runtime.h>

#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)

void check_hip_error(hipError_t error_code,const char* file, int line);

class HipProfiling{
    private:
    hipEvent_t memstart2D;
    hipEvent_t memstop2D;
    hipEvent_t memstart2H;
    hipEvent_t memstop2H;
    hipEvent_t computestart;
    hipEvent_t computestop;
    bool destroy = false;
    public:
    HipProfiling();
    ~HipProfiling();
    void begin_mem2D();
    void end_mem2D();
    void begin_mem2H();
    void end_mem2H();
    void begin_compute();
    void end_compute();
    KernelStats retreive(int repetitions); 
};


#endif

