#ifndef CL_ACCURACY_H
#define CL_ACCURACY_H
#include "accuracy.hpp"

class CLAccuracy : public IAccuracy {
public:
    KernelStats accuracy(const AccuracyData &aData, const AccuracySettings &aSettings, AccuracyResult &aResult) const override;
};

#endif