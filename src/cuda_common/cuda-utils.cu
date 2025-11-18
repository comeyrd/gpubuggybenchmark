#include "cuda-utils.hpp"
#include <cuda/std/chrono>

event_t **allocate_event_array(int count) {
    event_t **events = new event_t *[count];
    for (int i = 0; i < count; i++) {
        events[i] = new event_t;
        CHECK_CUDA(cudaEventCreate(&events[i]->native));
    }
    return events;
}

void free_event_array(event_t **events, int count) {
    for (int i = 0; i < count; i++) {
        CHECK_CUDA(cudaEventDestroy(events[i]->native));
        delete events[i];
    }
    delete[] events;
}

GpuStream::GpuStream() {
    stream = new stream_t;
    CHECK_CUDA(cudaStreamCreate(&stream->native));
}
GpuStream::~GpuStream() {
    CHECK_CUDA(cudaStreamDestroy(stream->native));
    delete stream;
}
bool GpuStream::get_stream_availability(){
    cudaError_t stream_status = cudaStreamQuery(stream->native);
    if(stream_status == cudaSuccess){
          return true;
    }else{
        std::cout << cudaGetErrorString(stream_status) << std::endl;
        return false;
    }
}
void GpuStream::synchronize(){
    CHECK_CUDA(cudaStreamSynchronize(stream->native));
}

void setup_gpu(int device) {
    std::cout <<"chosen device"<<device<<std::endl;
    CHECK_CUDA(cudaSetDevice(device));
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

GpuEventTimer::GpuEventTimer(ExecutionConfig config, stream_t *gpustream) : m_config(config), stream(gpustream) {
    memstart2D = new event_t;
    memstop2D = new event_t;
    memstart2H = new event_t;
    memstop2H = new event_t;
    CHECK_CUDA(cudaEventCreate(&memstart2D->native));
    CHECK_CUDA(cudaEventCreate(&memstop2D->native));
    CHECK_CUDA(cudaEventCreate(&memstart2H->native));
    CHECK_CUDA(cudaEventCreate(&memstop2H->native));
    warmupstart = allocate_event_array(m_config.m_warmups);
    warmupstop = allocate_event_array(m_config.m_warmups);
    repetitionstart = allocate_event_array(m_config.m_repetitions);
    repetitionstop = allocate_event_array(m_config.m_repetitions);
};

GpuEventTimer::~GpuEventTimer() {
    try {
        CHECK_CUDA(cudaEventDestroy(memstart2D->native));
        CHECK_CUDA(cudaEventDestroy(memstop2D->native));
        CHECK_CUDA(cudaEventDestroy(memstart2H->native));
        CHECK_CUDA(cudaEventDestroy(memstop2H->native));
        free_event_array(warmupstart, m_config.m_warmups);
        free_event_array(warmupstop, m_config.m_warmups);
        free_event_array(repetitionstart, m_config.m_repetitions);
        free_event_array(repetitionstop, m_config.m_repetitions);
    } catch (std::exception &e) {
        std::cerr << "Error destroying Cuda profiler" << e.what() << std::endl;
    }
};

void GpuEventTimer::begin_mem2D() {
    CHECK_CUDA(cudaEventRecord(memstart2D->native, stream->native));
};

void GpuEventTimer::end_mem2D() {
    CHECK_CUDA(cudaEventRecord(memstop2D->native, stream->native));
};

void GpuEventTimer::begin_mem2H() {
    CHECK_CUDA(cudaEventRecord(memstart2H->native, stream->native));
};

void GpuEventTimer::end_mem2H() {
    CHECK_CUDA(cudaEventRecord(memstop2H->native, stream->native));
};

void GpuEventTimer::begin_warmup() {
    CHECK_CUDA(cudaEventRecord(warmupstart[nb_w]->native, stream->native));
};
void GpuEventTimer::end_warmup() {
    CHECK_CUDA(cudaEventRecord(warmupstop[nb_w]->native, stream->native));
    nb_w++;
};
void GpuEventTimer::begin_repetition() {
    CHECK_CUDA(cudaEventRecord(repetitionstart[nb_r]->native, stream->native));
};
void GpuEventTimer::end_repetition() {
    CHECK_CUDA(cudaEventRecord(repetitionstop[nb_r]->native, stream->native));
    nb_r++;
};

KernelStats GpuEventTimer::retreive() {
    CHECK_CUDA(cudaEventSynchronize(memstop2D->native));
    CHECK_CUDA(cudaEventSynchronize(memstart2D->native));
    CHECK_CUDA(cudaEventSynchronize(memstart2H->native));
    CHECK_CUDA(cudaEventSynchronize(memstop2H->native));
    KernelStats stats(m_config);
    CHECK_CUDA(cudaEventElapsedTime(&stats.memcpy2D, memstart2D->native, memstop2D->native));
    CHECK_CUDA(cudaEventElapsedTime(&stats.memcpy2H, memstart2H->native, memstop2H->native));
    for (int w = 0; w < nb_w; w++) {
        CHECK_CUDA(cudaEventElapsedTime(&stats.warmup_duration[w], warmupstart[w]->native, warmupstop[w]->native));
    }
    for (int r = 0; r < nb_r; r++) {
        CHECK_CUDA(cudaEventElapsedTime(&stats.repetitions_duration[r], repetitionstart[r]->native, repetitionstop[r]->native));
    }
    stats.nb_r = nb_r;
    stats.nb_w = nb_w;
    return stats;
}

l2flushr::l2flushr() : cs() {
    int dev_id{};
    CHECK_CUDA(cudaGetDevice(&dev_id));
    CHECK_CUDA(cudaDeviceGetAttribute(&buffer_size, cudaDevAttrL2CacheSize, dev_id));
    if (buffer_size > 0) {
        void *buffer = l2_buffer;
        CHECK_CUDA(cudaMalloc(&buffer, static_cast<std::size_t>(buffer_size)));
        l2_buffer = reinterpret_cast<int *>(buffer);
    }
}
l2flushr::~l2flushr() {
    if (l2_buffer) {
        CHECK_CUDA(cudaFree(l2_buffer));
    }
}
void l2flushr::flush(stream_t *stream) {
    CHECK_CUDA(cudaMemsetAsync(l2_buffer, 0, static_cast<std::size_t>(buffer_size), stream->native));
}