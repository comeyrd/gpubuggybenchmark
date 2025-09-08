#include "hip-utils.hpp"
#include "gpu-utils.hpp"

void setup_gpu(){
    CHECK_HIP(hipSetDevice(0));
}

void reset_gpu(){
    CHECK_HIP(hipDeviceReset());
}

void check_hip_error(hipError_t error_code,const char* file, int line){
    if(error_code != hipSuccess){ 
        std::string msg = std::string("HIP Error : ") + hipGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

HipProfiling::HipProfiling(){
    CHECK_HIP(hipEventCreate(&memstart2D));
    CHECK_HIP(hipEventCreate(&memstop2D));
    CHECK_HIP(hipEventCreate(&memstart2H));
    CHECK_HIP(hipEventCreate(&memstop2H));
    CHECK_HIP(hipEventCreate(&computestart));
    CHECK_HIP(hipEventCreate(&computestop));
};

HipProfiling::~HipProfiling(){
    if(!destroy){
        try{
            CHECK_HIP(hipEventDestroy(memstart2D));
            CHECK_HIP(hipEventDestroy(memstop2D));
            CHECK_HIP(hipEventDestroy(memstart2H));
            CHECK_HIP(hipEventDestroy(memstop2H));
            CHECK_HIP(hipEventDestroy(computestart));
            CHECK_HIP(hipEventDestroy(computestop));
        }catch(std::exception &e){
            std::cerr << "Error destroying Hip profiler" << e.what()<<std::endl;
        }
    }
    
};

void HipProfiling::begin_mem2D(){
    CHECK_HIP(hipEventRecord(memstart2D));
};

void HipProfiling::end_mem2D(){
    CHECK_HIP(hipEventRecord(memstop2D));
};

void HipProfiling::begin_mem2H(){
    CHECK_HIP(hipEventRecord(memstart2H));
};

void HipProfiling::end_mem2H(){
    CHECK_HIP(hipEventRecord(memstop2H));
};


void HipProfiling::begin_compute(){
    CHECK_HIP(hipEventRecord(computestart));
};

void HipProfiling::end_compute(){
    CHECK_HIP(hipEventRecord(computestop));
};


KernelStats HipProfiling::retreive(){
    CHECK_HIP(hipEventSynchronize(memstop2D));
    CHECK_HIP(hipEventSynchronize(computestop));
    CHECK_HIP(hipEventSynchronize(memstop2H));

    KernelStats stats;
    CHECK_HIP(hipEventElapsedTime(&stats.memcpy2D, memstart2D, memstop2D));
    CHECK_HIP(hipEventElapsedTime(&stats.compute, computestart, computestop));
    CHECK_HIP(hipEventElapsedTime(&stats.memcpy2H, memstart2H, memstop2H));
    if(!destroy){ 
        CHECK_HIP(hipEventDestroy(memstart2D));
        CHECK_HIP(hipEventDestroy(memstop2D));
        CHECK_HIP(hipEventDestroy(memstart2H));
        CHECK_HIP(hipEventDestroy(memstop2H));
        CHECK_HIP(hipEventDestroy(computestart));
        CHECK_HIP(hipEventDestroy(computestop));
        destroy = true;
    }
   
    return stats;
}