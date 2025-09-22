#ifndef DR_ACCURACY_H
#define DR_ACCURACY_H
#include "accuracy.hpp"

class DRAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif