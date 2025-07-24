#ifndef DR_ACCURACY_H
#define DR_ACCURACY_H
#include "accuracy.hpp"

class DRAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif