#ifndef DG_BILATERAL_H
#define DG_BILATERAL_H
#include "bilateral.hpp"

class DGBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif