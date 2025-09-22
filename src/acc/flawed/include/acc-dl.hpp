#ifndef DL_ACCURACY_H
#define DL_ACCURACY_H
#include "accuracy.hpp"

class DLAccuracy : public IAccuracy {
public:
    KernelStats run(const AccuracyData &data, const AccuracySettings &settings, AccuracyResult &result) const override;
};

#endif