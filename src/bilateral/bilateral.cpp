#include "bilateral.hpp"
#include <cstring>
#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

template <int R>
void templated_run_cpu(const BilateralData &bData, BilateralData &bResult) {
#pragma omp parallel for collapse(2) 
    for (uint idx = 0; idx < bData.width; idx++) {
        for (uint idy = 0; idy < bData.height; idy++) {

            uint id = idy * bData.width + idx;
            float I = bData.image[id];
            float res = 0.f;
            float normalization = 0.f;

// window centered at the coordinate (idx, idy)
            for (int i = -R; i <= R; i++) {
                for (int j = -R; j <= R; j++) {

                    uint idk = idx + i;
                    uint idl = idy + j;

                    // mirror edges
                    if (idk < 0)
                        idk = -idk;
                    if (idl < 0)
                        idl = -idl;
                    if (idk > bData.width - 1)
                        idk = bData.width - 1 - i;
                    if (idl > bData.height - 1)
                        idl = bData.height - 1 - j;

                    uint id_w = idl * bData.width + idk;
                    float I_w = bData.image[id_w];

                    // range kernel for smoothing differences in intensities
                    float range = -(I - I_w) * (I - I_w) / (2.f * bData.variance_I);

                    // spatial kernel for smoothing differences in coordinates
                    float spatial = -((idk - idx) * (idk - idx) + (idl - idy) * (idl - idy)) / (2.f * bData.variance_spatial);

                    // combined weight
                    float weight = bData.a_square * expf(spatial + range);
                    normalization += weight;
                    res += (I_w * weight);
                }
            }
            bResult.image[id] = res / normalization;
        }
    }
}
#pragma GCC pop_options

void BilateralData::generate_random() {
    srand(123);
    for (uint i = 0; i < this->size; i++)
        this->image[i] = rand() % 256;

}

REGISTER_CLASS(I_IKernel, Bilateral);