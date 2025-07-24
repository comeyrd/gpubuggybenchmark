#ifndef MO_ACCURACY_H
#define MO_ACCURACY_H
#include "accuracy.hpp"

class MOAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif