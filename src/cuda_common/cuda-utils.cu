#include "cuda-utils.hpp"
#include "gpu-utils.hpp"

void setup_gpu() {
    CHECK_CUDA(cudaSetDevice(0));
}

void reset_gpu() {
    CHECK_CUDA(cudaDeviceReset());
    // setup_gpu();
}

void check_cuda_error(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

CudaProfiling::CudaProfiling(BaseSettings settings_) : settings(settings_) {
    CHECK_CUDA(cudaEventCreate(&memstart2D));
    CHECK_CUDA(cudaEventCreate(&memstop2D));
    CHECK_CUDA(cudaEventCreate(&memstart2H));
    CHECK_CUDA(cudaEventCreate(&memstop2H));
    warmupstart = new cudaEvent_t[settings.warmup];
    warmupstop = new cudaEvent_t[settings.warmup];
    repetitionstart = new cudaEvent_t[settings.repetitions];
    repetitionstop = new cudaEvent_t[settings.repetitions];
    for (int w = 0; w < settings.warmup; w++) {
        CHECK_CUDA(cudaEventCreate(&warmupstart[w]));
        CHECK_CUDA(cudaEventCreate(&warmupstop[w]));
    }
    for (int r = 0; r < settings.repetitions; r++) {
        CHECK_CUDA(cudaEventCreate(&repetitionstart[r]));
        CHECK_CUDA(cudaEventCreate(&repetitionstop[r]));
    }
};

CudaProfiling::~CudaProfiling() {
    if (!destroy) {
        try {
            CHECK_CUDA(cudaEventDestroy(memstart2D));
            CHECK_CUDA(cudaEventDestroy(memstop2D));
            CHECK_CUDA(cudaEventDestroy(memstart2H));
            CHECK_CUDA(cudaEventDestroy(memstop2H));
            for (int w = 0; w < settings.warmup; w++) {
                CHECK_CUDA(cudaEventDestroy(warmupstart[w]));
                CHECK_CUDA(cudaEventDestroy(warmupstop[w]));
            }
            for (int r = 0; r < settings.repetitions; r++) {
                CHECK_CUDA(cudaEventDestroy(repetitionstart[r]));
                CHECK_CUDA(cudaEventDestroy(repetitionstop[r]));
            }
        } catch (std::exception &e) {
            // std::cerr << "Error destroying Cuda profiler" << e.what()<<std::endl;
        }
    }
};

void CudaProfiling::begin_mem2D() {
    CHECK_CUDA(cudaEventRecord(memstart2D));
};

void CudaProfiling::end_mem2D() {
    CHECK_CUDA(cudaEventRecord(memstop2D));
};

void CudaProfiling::begin_mem2H() {
    CHECK_CUDA(cudaEventRecord(memstart2H));
};

void CudaProfiling::end_mem2H() {
    CHECK_CUDA(cudaEventRecord(memstop2H));
};

void CudaProfiling::begin_warmup() {
    CHECK_CUDA(cudaEventRecord(warmupstart[nb_w]));
};
void CudaProfiling::end_warmup() {
    CHECK_CUDA(cudaEventRecord(warmupstop[nb_w]));
    nb_w++;
};
void CudaProfiling::begin_repetition() {
    CHECK_CUDA(cudaEventRecord(repetitionstart[nb_r]));
};
void CudaProfiling::end_repetition() {
    CHECK_CUDA(cudaEventRecord(repetitionstop[nb_r]));
    nb_r++;
};

KernelStats CudaProfiling::retreive() {
    CHECK_CUDA(cudaEventSynchronize(memstop2D));
    CHECK_CUDA(cudaEventSynchronize(memstop2H));

    KernelStats stats(settings);
    CHECK_CUDA(cudaEventElapsedTime(&stats.memcpy2D, memstart2D, memstop2D));
    CHECK_CUDA(cudaEventElapsedTime(&stats.memcpy2H, memstart2H, memstop2H));
    for (int w = 0; w < nb_w; w++) {
        CHECK_CUDA(cudaEventElapsedTime(&stats.warmup_duration[w], warmupstart[w], warmupstop[w]));
    }
    for (int r = 0; r < nb_r; r++) {
        CHECK_CUDA(cudaEventElapsedTime(&stats.repetitions_duration[r], repetitionstart[r], repetitionstop[r]));
    }
    stats.nb_r = nb_r;
    stats.nb_w = nb_w;
    if (!destroy) {
        CHECK_CUDA(cudaEventDestroy(memstart2D));
        CHECK_CUDA(cudaEventDestroy(memstop2D));
        CHECK_CUDA(cudaEventDestroy(memstart2H));
        CHECK_CUDA(cudaEventDestroy(memstop2H));
        for (int w = 0; w < settings.warmup; w++) {
            CHECK_CUDA(cudaEventDestroy(warmupstart[w]));
            CHECK_CUDA(cudaEventDestroy(warmupstop[w]));
        }
        for (int r = 0; r < settings.repetitions; r++) {
            CHECK_CUDA(cudaEventDestroy(repetitionstart[r]));
            CHECK_CUDA(cudaEventDestroy(repetitionstop[r]));
        }
        destroy = true;
    }
    return stats;
}