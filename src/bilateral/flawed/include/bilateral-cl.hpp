#ifndef CL_BILATERAL_H
#define CL_BILATERAL_H
#include "bilateral.hpp"

class CLBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif