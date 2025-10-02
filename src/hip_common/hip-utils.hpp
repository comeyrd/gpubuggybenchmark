#ifndef HIP_UTILS_HPP  
#define  HIP_UTILS_HPP
#include "Kernel.hpp"
#include "hip/hip_runtime.h"

#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)

void check_hip_error(hipError_t error_code,const char* file, int line);

class HipProfiling{
    private:
    hipEvent_t memstart2D;
    hipEvent_t memstop2D;
    hipEvent_t memstart2H;
    hipEvent_t memstop2H;
    hipEvent_t* warmupstart;
    hipEvent_t* warmupstop;
    hipEvent_t* repetitionstart;
    hipEvent_t* repetitionstop;
    BaseSettings settings;
    bool destroy = false;
    int nb_w = 0;//what warmup iteration are we on
    int nb_r = 0;//what repetition iteration are we on
    public:
    HipProfiling(BaseSettings settings_);
    ~HipProfiling();
    void begin_mem2D();
    void end_mem2D();
    void begin_mem2H();
    void end_mem2H();
    void begin_warmup();
    void end_warmup();
    void begin_repetition();
    void end_repetition();
    KernelStats retreive(); 
};

//TODO Update the tools to use "nsys" and "ncu" and other.

#endif

