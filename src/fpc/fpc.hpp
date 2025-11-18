#ifndef FPC_H
#define FPC_H
#include "Kernel.hpp"
#include "Manager.hpp"
#include "gpu-utils.hpp"
#include "version.hpp"
#include <memory>
#include <string>
#include <unordered_map>

constexpr int WORK_GROUP_SIZE = 256;
constexpr size_t MINIMAL_LENGTH = WORK_GROUP_SIZE * 10000; // 250M

struct FPCData : IData {
    size_t length;
    size_t b_size;
    int wgz;
    ulong* values;
    
    explicit FPCData(const int& work_size) : IData(work_size) {
        length = MINIMAL_LENGTH * work_size;
        wgz = WORK_GROUP_SIZE;
        b_size = length * sizeof(ulong);
        
        values = (ulong *)malloc(b_size);
        if (!values) throw std::bad_alloc();
    }
    
    void generate_random() override;
    
    void resize(int work_size) override {
        if (values) {
            free(values);
            values = nullptr;
        }
        
        length = MINIMAL_LENGTH * work_size;
        wgz = WORK_GROUP_SIZE;
        b_size = length * sizeof(ulong);
        
        values = (ulong *)malloc(b_size);
        if (!values) throw std::bad_alloc();
    }
    
    ~FPCData() {
        free(values);
    }
};

struct FPCResult : IResult {
    unsigned size_;
    
    explicit FPCResult(const int& work_size) : IResult(work_size) {
        size_ = 0;
    }
    
    void resize(int work_size) override {
        // nothing to do
    }
    
    bool operator==(const FPCResult& other) const {
        return other.size_ == this->size_;
    }
};

inline std::ostream& operator<<(std::ostream& os, FPCResult &result) {
    os << result.size_;
    return os;
}

using IFPC = IVersion<FPCData, FPCResult>;

class FPC : public IKernel<FPCData, FPCResult> {
public:
    FPC() = default;
    
    std::string name() override {
        return "FPC";
    }
    
private:
    void run_cpu() override;
};

#endif