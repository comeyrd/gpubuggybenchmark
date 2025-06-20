#ifndef FPCREFERENCE
#define FPCREFERENCE
#include "fpc.hpp"

void fpc_reference(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs);
void fpc2_reference(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs);
unsigned fpc_cpu(ulong* values, unsigned size);

class ReferenceFpc : public IFpc{
    public:
    void fpc(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs) const override{
        return fpc_reference(values,cmp_size_hw,values_size,wgs);
    }
    void fpc2(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs) const override{
        return fpc2_reference(values,cmp_size_hw,values_size,wgs);
    }
};
#endif