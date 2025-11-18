#ifndef ACCURACY_H
#define ACCURACY_H
#include "Kernel.hpp"
#include "Manager.hpp"
#include "gpu-utils.hpp"
#include "version.hpp"
#include <memory>
#include <string>
#include <unordered_map>

#define GPU_NUM_THREADS 256

constexpr int MINIMAL_NROWS = 10000;
constexpr int MINIMAL_NDIMS = 1000;
constexpr int TOP_K = 10;

struct AccuracyData : IData {
    int n_rows;
    int ndims;
    int topk;
    
    int *label;
    size_t label_sz_bytes;
    float *data;
    size_t data_sz_bytes;
    
    explicit AccuracyData(const int& work_size) : IData(work_size) {
        n_rows = MINIMAL_NROWS * work_size;
        ndims = MINIMAL_NDIMS * work_size;
        topk = TOP_K;
        
        const int data_size = n_rows * ndims;
        label_sz_bytes = n_rows * sizeof(int);
        data_sz_bytes = data_size * sizeof(float);
        
        label = (int *)malloc(label_sz_bytes);
        data = (float *)malloc(data_sz_bytes);
        
        if (!data || !label) throw std::bad_alloc();
    }
    
    void generate_random() override;
    
    void resize(int work_size) override {
        if (label) {
            free(label);
            label = nullptr;
        }
        if (data) {
            free(data);
            data = nullptr;
        }
        
        n_rows = MINIMAL_NROWS * work_size;
        ndims = MINIMAL_NDIMS;
        topk = TOP_K;
        
        const int data_size = n_rows * ndims;
        label_sz_bytes = n_rows * sizeof(int);
        data_sz_bytes = data_size * sizeof(float);
        
        label = (int *)malloc(label_sz_bytes);
        data = (float *)malloc(data_sz_bytes);
        
        if (!data || !label) throw std::bad_alloc();
    }
    
    ~AccuracyData() {
        free(label);
        free(data);
    }
};

struct AccuracyResult : IResult {
    int count;
    
    explicit AccuracyResult(const int& work_size) : IResult(work_size) {
        count = 0;
    }
    void resize(int work_size) override{
        //nothing to do 
    };
    bool operator==(const AccuracyResult& other) const {
        return other.count == this->count;
    }
};

inline std::ostream& operator<<(std::ostream& os, AccuracyResult &a_result) {
    os << a_result.count;
    return os;
}

using IAccuracy = IVersion<AccuracyData, AccuracyResult>;

class Accuracy : public IKernel<AccuracyData, AccuracyResult> {
public:
    Accuracy() = default;
    
    std::string name() override {
        return "Accuracy";
    }
    
private:
    void run_cpu() override;
};

#endif