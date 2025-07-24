#include "cuda-utils.hpp"
#include "gpu-utils.hpp"

void setup_gpu(){
    CHECK_CUDA(cudaSetDevice(0));
}

void reset_gpu(){
    CHECK_CUDA(cudaDeviceReset());
}

void check_cuda_error(cudaError_t error_code,const char* file, int line){
    if(error_code != cudaSuccess){ 
        std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

CudaProfiling::CudaProfiling(){
    CHECK_CUDA(cudaEventCreate(&memstart2D));
    CHECK_CUDA(cudaEventCreate(&memstop2D));
    CHECK_CUDA(cudaEventCreate(&memstart2H));
    CHECK_CUDA(cudaEventCreate(&memstop2H));
    CHECK_CUDA(cudaEventCreate(&computestart));
    CHECK_CUDA(cudaEventCreate(&computestop));
};

CudaProfiling::~CudaProfiling(){
    if(!destroy){
        try{
            CHECK_CUDA(cudaEventDestroy(memstart2D));
            CHECK_CUDA(cudaEventDestroy(memstop2D));
            CHECK_CUDA(cudaEventDestroy(memstart2H));
            CHECK_CUDA(cudaEventDestroy(memstop2H));
            CHECK_CUDA(cudaEventDestroy(computestart));
            CHECK_CUDA(cudaEventDestroy(computestop));
        }catch(std::exception &e){
            std::cerr << "Error destroying Cuda profiler" << e.what()<<std::endl;
        }
    }
    
};

void CudaProfiling::begin_mem2D(){
    CHECK_CUDA(cudaEventRecord(memstart2D));
};

void CudaProfiling::end_mem2D(){
    CHECK_CUDA(cudaEventRecord(memstop2D));
};

void CudaProfiling::begin_mem2H(){
    CHECK_CUDA(cudaEventRecord(memstart2H));
};

void CudaProfiling::end_mem2H(){
    CHECK_CUDA(cudaEventRecord(memstop2H));
};


void CudaProfiling::begin_compute(){
    CHECK_CUDA(cudaEventRecord(computestart));
};

void CudaProfiling::end_compute(){
    CHECK_CUDA(cudaEventRecord(computestop));
};


KernelStats CudaProfiling::retreive(){
    CHECK_CUDA(cudaEventSynchronize(memstop2D));
    CHECK_CUDA(cudaEventSynchronize(computestop));
    CHECK_CUDA(cudaEventSynchronize(memstop2H));

    KernelStats stats;
    CHECK_CUDA(cudaEventElapsedTime(&stats.memcpy2D, memstart2D, memstop2D));
    CHECK_CUDA(cudaEventElapsedTime(&stats.compute, computestart, computestop));
    CHECK_CUDA(cudaEventElapsedTime(&stats.memcpy2H, memstart2H, memstop2H));
    if(!destroy){ 
        CHECK_CUDA(cudaEventDestroy(memstart2D));
        CHECK_CUDA(cudaEventDestroy(memstop2D));
        CHECK_CUDA(cudaEventDestroy(memstart2H));
        CHECK_CUDA(cudaEventDestroy(memstop2H));
        CHECK_CUDA(cudaEventDestroy(computestart));
        CHECK_CUDA(cudaEventDestroy(computestop));
        destroy = true;
    }
   
    return stats;
}