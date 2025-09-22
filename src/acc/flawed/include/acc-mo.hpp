#ifndef MO_ACCURACY_H
#define MO_ACCURACY_H
#include "accuracy.hpp"

class MOAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif