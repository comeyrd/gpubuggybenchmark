#ifndef ML_ACCURACY_H
#define ML_ACCURACY_H
#include "accuracy.hpp"

class MLAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif