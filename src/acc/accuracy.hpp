#ifndef ACCURACY_H
#define ACCURACY_H
#include "Kernel.hpp"
#include "Manager.hpp"
#include "version.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <forward_list>
#define GPU_NUM_THREADS 256

constexpr int NROWS = 2000000;
constexpr int NDIMS = 1000;
constexpr int TOP_K = 10;


const std::string CSV_FILE = "./csv/acuracy_";

struct AccuracySettings {
   //int block_nbr; Problem with cubreduce because the blocksize needs to be known at compile time.
    int n_rows = NROWS;
    int ndims = NDIMS;
    int topk = TOP_K;
    int repetitions;
    int warmup;
    int grid_sz = (NROWS + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;
    AccuracySettings(int n_rows_,int ndims_, int topk_, int repetitions_, int warmup_):n_rows(n_rows_),ndims(ndims_), topk(topk_), repetitions(repetitions_), warmup(warmup_),
          grid_sz((n_rows_ + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS){};
    AccuracySettings(int repetitions_, int warmup_):repetitions(repetitions_),warmup(warmup_){};
};

struct AccuracyData {
    int n_rows;
    int ndims;
    int *label;
    size_t label_sz_bytes;
    float *data;
    size_t data_sz_bytes;
    int topk;
    explicit AccuracyData(const AccuracySettings& settings){
        n_rows = settings.n_rows;
        ndims = settings.ndims;
        topk = settings.topk;
        const int data_size = n_rows * ndims;
        label_sz_bytes = n_rows * sizeof(int);
        data_sz_bytes = data_size * sizeof(float);
        label = (int *)malloc(label_sz_bytes);
        data = (float *)malloc(data_sz_bytes);
        if (!data) {
            perror("malloc failed for data");
            exit(EXIT_FAILURE);
        }
        if (!label) {
            perror("malloc failed for label");
            exit(EXIT_FAILURE);
        }
    }

    void generate_random();

    ~AccuracyData(){
        free(label);
        free(data);
    }
    //Making the thing not copiable etc etc
    AccuracyData(const AccuracyData&) = delete;
    AccuracyData& operator=(const AccuracyData&) = delete;
    AccuracyData(AccuracyData&&) noexcept = default;
    AccuracyData& operator=(AccuracyData&&) noexcept = default;
};


struct AccuracyResult {
    int count;
    bool operator==(const AccuracyResult& other) const {
    return other.count == this->count;
};
    explicit AccuracyResult(const AccuracySettings &settings){ 
        count = 0;
    };
};

inline std::ostream& operator<<(std::ostream& os,AccuracyResult a_result) {
    os << a_result.count ;
    return os;
};


using IAccuracy = IVersion<AccuracyData,AccuracySettings,AccuracyResult>;

class Accuracy : public IKernel<AccuracyData, AccuracySettings, AccuracyResult >{

    public :
        Accuracy() = default;
        std::string name() override{
            return "Accuracy";
        }
    private:
        void run_cpu() override;
};

#endif