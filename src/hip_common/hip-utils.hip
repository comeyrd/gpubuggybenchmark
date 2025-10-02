#include "hip/hip_runtime.h"
#include "hip-utils.hpp"
#include "gpu-utils.hpp"

void setup_gpu() {
    CHECK_HIP(hipSetDevice(0));
}

void reset_gpu() {
    CHECK_HIP(hipDeviceReset());
    // setup_gpu();
}

void check_hip_error(hipError_t error_code, const char *file, int line) {
    if (error_code != hipSuccess) {
        std::string msg = std::string("Hip Error : ") + hipGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

HipProfiling::HipProfiling(BaseSettings settings_) : settings(settings_) {
    CHECK_HIP(hipEventCreate(&memstart2D));
    CHECK_HIP(hipEventCreate(&memstop2D));
    CHECK_HIP(hipEventCreate(&memstart2H));
    CHECK_HIP(hipEventCreate(&memstop2H));
    warmupstart = new hipEvent_t[settings.warmup];
    warmupstop = new hipEvent_t[settings.warmup];
    repetitionstart = new hipEvent_t[settings.repetitions];
    repetitionstop = new hipEvent_t[settings.repetitions];
    for (int w = 0; w < settings.warmup; w++) {
        CHECK_HIP(hipEventCreate(&warmupstart[w]));
        CHECK_HIP(hipEventCreate(&warmupstop[w]));
    }
    for (int r = 0; r < settings.repetitions; r++) {
        CHECK_HIP(hipEventCreate(&repetitionstart[r]));
        CHECK_HIP(hipEventCreate(&repetitionstop[r]));
    }
};

HipProfiling::~HipProfiling() {
    if (!destroy) {
        try {
            CHECK_HIP(hipEventDestroy(memstart2D));
            CHECK_HIP(hipEventDestroy(memstop2D));
            CHECK_HIP(hipEventDestroy(memstart2H));
            CHECK_HIP(hipEventDestroy(memstop2H));
            for (int w = 0; w < settings.warmup; w++) {
                CHECK_HIP(hipEventDestroy(warmupstart[w]));
                CHECK_HIP(hipEventDestroy(warmupstop[w]));
            }
            for (int r = 0; r < settings.repetitions; r++) {
                CHECK_HIP(hipEventDestroy(repetitionstart[r]));
                CHECK_HIP(hipEventDestroy(repetitionstop[r]));
            }
        } catch (std::exception &e) {
            // std::cerr << "Error destroying Cuda profiler" << e.what()<<std::endl;
        }
    }
};

void HipProfiling::begin_mem2D() {
    CHECK_HIP(hipEventRecord(memstart2D));
};

void HipProfiling::end_mem2D() {
    CHECK_HIP(hipEventRecord(memstop2D));
};

void HipProfiling::begin_mem2H() {
    CHECK_HIP(hipEventRecord(memstart2H));
};

void HipProfiling::end_mem2H() {
    CHECK_HIP(hipEventRecord(memstop2H));
};

void HipProfiling::begin_warmup() {
    CHECK_HIP(hipEventRecord(warmupstart[nb_w]));
};
void HipProfiling::end_warmup() {
    CHECK_HIP(hipEventRecord(warmupstop[nb_w]));
    nb_w++;
};
void HipProfiling::begin_repetition() {
    CHECK_HIP(hipEventRecord(repetitionstart[nb_r]));
};
void HipProfiling::end_repetition() {
    CHECK_HIP(hipEventRecord(repetitionstop[nb_r]));
    nb_r++;
};

KernelStats HipProfiling::retreive() {
    CHECK_HIP(hipEventSynchronize(memstop2D));
    CHECK_HIP(hipEventSynchronize(memstop2H));

    KernelStats stats(settings);
    CHECK_HIP(hipEventElapsedTime(&stats.memcpy2D, memstart2D, memstop2D));
    CHECK_HIP(hipEventElapsedTime(&stats.memcpy2H, memstart2H, memstop2H));
    for (int w = 0; w < nb_w; w++) {
        CHECK_HIP(hipEventElapsedTime(&stats.warmup_duration[w], warmupstart[w], warmupstop[w]));
    }
    for (int r = 0; r < nb_r; r++) {
        CHECK_HIP(hipEventElapsedTime(&stats.repetitions_duration[r], repetitionstart[r], repetitionstop[r]));
    }
    stats.nb_r = nb_r;
    stats.nb_w = nb_w;
    if (!destroy) {
        CHECK_HIP(hipEventDestroy(memstart2D));
        CHECK_HIP(hipEventDestroy(memstop2D));
        CHECK_HIP(hipEventDestroy(memstart2H));
        CHECK_HIP(hipEventDestroy(memstop2H));
        for (int w = 0; w < settings.warmup; w++) {
            CHECK_HIP(hipEventDestroy(warmupstart[w]));
            CHECK_HIP(hipEventDestroy(warmupstop[w]));
        }
        for (int r = 0; r < settings.repetitions; r++) {
            CHECK_HIP(hipEventDestroy(repetitionstart[r]));
            CHECK_HIP(hipEventDestroy(repetitionstop[r]));
        }
        destroy = true;
    }
    return stats;
}
