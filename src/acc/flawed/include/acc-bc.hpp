#ifndef BC_ACCURACY_H
#define BC_ACCURACY_H
#include "accuracy.hpp"

class BCAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif