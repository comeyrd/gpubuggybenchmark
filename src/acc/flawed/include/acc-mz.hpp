#ifndef MZ_ACCURACY_H
#define MZ_ACCURACY_H
#include "accuracy.hpp"

class MZAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif