#include "accuracy.hpp"
#include "gpu-utils.hpp"
#include <random>

void Accuracy::run_cpu() {
    m_cpu_result.count = 0;
    for (int row = 0; row < m_data.n_rows; row++) {
        const int label = m_data.label[row];
        const float label_pred = m_data.data[row * m_data.ndims + label];
        int ngt = 0;
        for (int col = 0; col < m_data.ndims; col++) {
            const float pred = m_data.data[row * m_data.ndims + col];
            if (pred > label_pred || (pred == label_pred && col <= label)) {
                ++ngt;
            }
        }
        if (ngt <= m_data.topk) {
            ++m_cpu_result.count;
        }
    }
}

void AccuracyData::generate_random() {
    srand(123);
    for (int i = 0; i < n_rows; i++)
        label[i] = rand() % ndims;

    std::default_random_engine g(123);
    std::uniform_real_distribution<float> distr(0.f, 1.f);
    for (int i = 0; i < n_rows * ndims; i++) {
        data[i] = distr(g);
    }
}

REGISTER_CLASS(I_IKernel, Accuracy);
