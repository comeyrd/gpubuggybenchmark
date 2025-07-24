#ifndef MZ_ACCURACY_H
#define MZ_ACCURACY_H
#include "accuracy.hpp"

class MZAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif