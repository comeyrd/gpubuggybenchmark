#ifndef MA_ACCURACY_H
#define MA_ACCURACY_H
#include "accuracy.hpp"

class MAAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif