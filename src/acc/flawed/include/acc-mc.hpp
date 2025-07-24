#ifndef MC_ACCURACY_H
#define MC_ACCURACY_H
#include "accuracy.hpp"

class MCAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif