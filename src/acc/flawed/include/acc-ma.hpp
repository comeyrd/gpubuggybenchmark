#ifndef MA_ACCURACY_H
#define MA_ACCURACY_H
#include "accuracy.hpp"

class MAAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif