#ifndef BILATERAL_H
#define BILATERAL_H
#include "Kernel.hpp"
#include "Manager.hpp"
#include "version.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include "gpu-utils.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

struct BilateralData {
    uint width;
    uint height;
    size_t b_size;
    uint size;
    float* inputImage; 
};

constexpr float VARIANCE_I = 10.0f;
constexpr float VARIANCE_SPATIALE = 12.0f;
constexpr float A_SQUARE = 0.5f / (VARIANCE_I * (float)M_PI);

constexpr int WIDTH = 1920;
constexpr int HEIGHT = 1080;

struct BilateralSettings {
    int width;
    int height;
    float a_square;      
    float variance_I;    
    float variance_spatial; 
};


struct BilateralResult {
    float* outputImage;
    int size;
    size_t b_size;
    uint32_t checksum; 

    uint32_t computeChecksum() const;

    void updateChecksum() {
        checksum = computeChecksum();
    }
    bool operator==(const BilateralResult& other) const {
        return size == other.size && checksum == other.checksum;
    }
};


class IBilateral {
public:
    virtual ~IBilateral() = default;
    virtual KernelStats bilateral(const BilateralData &bData, const BilateralSettings &bSettings, BilateralResult &bResult) const = 0;
};

class Bilateral : public IKernel {
public:
    int run_kernel(int argc, char **argv);

private:
    void register_cli_options(argparse::ArgumentParser &parser) override;
    std::vector<std::string> list_versions() override;
    void run_versions(class_umap<IBilateral> versions, int repetitions, int warmup, BilateralData bData, BilateralSettings bSettings);
    KernelStats run_impl(std::shared_ptr<IBilateral> bilateral_impl, int repetitions, int warmup, BilateralData &bData, BilateralSettings &bSettings,BilateralResult &bResult);
    template <int R> void run_cpu(const BilateralData &bData, const BilateralSettings &bSettings, BilateralResult &bResult);
    BilateralData random_data(const BilateralSettings &bSettings);
};

#endif