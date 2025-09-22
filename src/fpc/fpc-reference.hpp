#ifndef FPCREFERENCE
#define FPCREFERENCE
#include "fpc.hpp"

class ReferenceFPC : public IFPC {
public:
    KernelStats run(const FPCData &data, const FPCSettings &settings, FPCResult &result) const override;
};

class ReferenceFPC2 : public IFPC {
public:
    KernelStats run(const FPCData &data, const FPCSettings &settings, FPCResult &result) const override;
};

#endif