#ifndef BILATERAL_H
#define BILATERAL_H
#include "Kernel.hpp"
#include "Manager.hpp"
#include "gpu-utils.hpp"
#include "version.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#define _USE_MATH_DEFINES
#include <cmath>


constexpr float VARIANCE_I = 10.0f;
constexpr float VARIANCE_SPATIALE = 12.0f;
constexpr float A_SQUARE = 0.5f / (VARIANCE_I * (float)M_PI);

constexpr int WIDTH = 4000;
constexpr int HEIGHT = 4000;

constexpr float ROUNDING_ERROR = 1e-3;

struct BilateralSettings {
    int repetitions;
    int warmup;
    int width = WIDTH;
    int height = HEIGHT;
    float a_square = A_SQUARE;
    float variance_I = VARIANCE_I;
    float variance_spatial = VARIANCE_SPATIALE;
    BilateralSettings(int repetitions_, int warmup_):repetitions(repetitions_),warmup(warmup_){};
};

struct BilateralData {
    uint width;
    uint height;
    size_t b_size;
    uint size;
    float *image;

    explicit BilateralData(const BilateralSettings &settings){
        width = settings.width;
        height = settings.height;
        size = settings.width * settings.height;
        b_size = size * sizeof(float);
        image = (float *)malloc(b_size);
        if (!image) throw std::bad_alloc();

    };

    void generate_random();

    ~BilateralData(){
        free(image);
    }
    bool operator==(const BilateralData &other) const {
        if (size == other.size) {
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (fabsf(image[i] - other.image[i]) > ROUNDING_ERROR) {
                    count ++;
                }
            }
            if(count == 0){
                return true;
            }else{
                std::cout << count<<std::endl;
                return false;
            }
        } else {
            return false;
        }
    }
    //Making the thing not copiable etc etc
    BilateralData(const BilateralData&) = delete;
    BilateralData& operator=(const BilateralData&) = delete;
    BilateralData(BilateralData&&) noexcept = default;
    BilateralData& operator=(BilateralData&&) noexcept = default;
};

inline std::ostream& operator<<(std::ostream& os , BilateralData &a_result) {
    os << "-" ;
    return os;
};



using IBilateral = IVersion<BilateralData,BilateralSettings,BilateralData>;

template <int R>
void templated_run_cpu(const BilateralData &bData, const BilateralSettings &bSettings, BilateralData &bResult);


class Bilateral : public IKernel<BilateralData, BilateralSettings, BilateralData >{

    public :
        Bilateral() = default;
        std::string name() override{
            return "Bilateral";
        }
    private:
        void run_cpu() override{
            return templated_run_cpu<4>(data,settings,cpu_result);
        };
};



#endif