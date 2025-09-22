#ifndef FPC_H
#define FPC_H
#include "Kernel.hpp"
#include "Manager.hpp"
#include "gpu-utils.hpp"
#include "version.hpp"
#include <memory>
#include <string>
#include <unordered_map>

constexpr uint _WORK_GROUP_SIZE = 256;
constexpr size_t _LENGTH = _WORK_GROUP_SIZE * 4500000;//250M
struct FPCSettings {
    int repetitions;
    int warmup;
    size_t length = _LENGTH;
    uint wgz = _WORK_GROUP_SIZE;
    FPCSettings(int repetitions_, int warmup_):repetitions(repetitions_),warmup(warmup_){};
};

struct FPCData {
    size_t length;
    size_t b_size;
    ulong* values;

    explicit FPCData(const FPCSettings &settings){
        length = settings.length;
        b_size = length * sizeof(ulong);
        values = (ulong *)malloc(b_size);
        if (!values) throw std::bad_alloc();
    };

    void generate_random();

    ~FPCData(){
        free(values);
    }
    //Making the thing not copiable etc etc
    FPCData(const FPCData&) = delete;
    FPCData& operator=(const FPCData&) = delete;
    FPCData(FPCData&&) noexcept = default;
    FPCData& operator=(FPCData&&) noexcept = default;
};

struct FPCResult {
    unsigned size_;
    bool operator==(const FPCResult& other) const {
    return other.size_ == this->size_;
};
    explicit FPCResult(const FPCSettings &settings){ 
        size_ = 0;
    };
};

inline std::ostream& operator<<(std::ostream& os,const FPCResult &result) {
    os << result.size_ ;
    return os;
};


using IFPC = IVersion<FPCData,FPCSettings,FPCResult>;


class FPC : public IKernel<FPCData, FPCSettings, FPCResult >{

    public :
        FPC() = default;
        std::string name() override{
            return "FPC";
        }
    private:
        void run_cpu() override;
};
#endif
