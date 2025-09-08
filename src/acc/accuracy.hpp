#ifndef ACCURACY_H
#define ACCURACY_H
#include "Kernel.hpp"
#include "Manager.hpp"
#include "version.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <forward_list>

//TODO fix the generation of random data, or use a dataset ?

constexpr int NROWS = 1000;
constexpr int NDIMS = 50;
constexpr int TOP_K = 200;

constexpr int GRID_SZ = NROWS / 2;

const std::string CSV_FILE = "./csv/acuracy_";

struct AccuracyData {
    int n_rows;
    int ndims;
    int *label;
    size_t label_sz_bytes;
    float *data;
    size_t data_sz_bytes;
    int topk;
};

struct AccuracySettings {
   //int block_nbr; Problem with cubreduce because the blocksize needs to be known at compile time.
    int grid_sz;
};

struct AccuracyResult {
    int count;
    bool operator==(const AccuracyResult& other) const {
    return other.count == this->count;
};
};

class IAccuracy {
public:
    virtual ~IAccuracy() = default;
    virtual KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const = 0;
};

class Accuracy : public IKernel {
public:
    int run_kernel(int argc, char **argv) override;

private:
    void register_cli_options(argparse::ArgumentParser &parser) override;
    std::vector<std::string> list_versions() override;
    void run_versions(class_umap<IAccuracy> versions, int repetitions, int warmup);
    std::forward_list<KernelStats> run_impl(std::shared_ptr<IAccuracy> accuracy_impl, int repetitions, int warmup, AccuracyData &aData, AccuracySettings &aSettings,AccuracyResult &aResult);
    int run_cpu(const AccuracyData &aData);
    AccuracyData random_data(int n_rows, int ndims, int top_k);
};

#endif