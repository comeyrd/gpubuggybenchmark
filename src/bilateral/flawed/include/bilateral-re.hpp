#ifndef RE_BILATERAL_H
#define RE_BILATERAL_H
#include "bilateral.hpp"

class REBilateral : public IBilateral {
public:
    KernelStats run(const BilateralData &data, const BilateralSettings &settings, BilateralData &result) const override;
};

#endif