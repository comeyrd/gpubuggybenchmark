
#ifndef CUDAPROFILINGHPP
#define CUDAPROFILINGHPP
#include "cuda-utils.hpp"
class CudaProfiling {
private:
    cudaEvent_t memstart2D;
    cudaEvent_t memstop2D;
    cudaEvent_t memstart2H;
    cudaEvent_t memstop2H;
    cudaEvent_t *warmupstart;
    cudaEvent_t *warmupstop;
    cudaEvent_t *repetitionstart;
    cudaEvent_t *repetitionstop;
    BaseSettings settings;
    cudaStream_t cs;
    bool destroy = false;
    int nb_w = 0; // what warmup iteration are we on
    int nb_r = 0; // what repetition iteration are we on
public:
    CudaProfiling(BaseSettings settings_,cudaStream_t cudastream);
    ~CudaProfiling();
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

#endif