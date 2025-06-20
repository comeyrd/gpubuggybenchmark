#ifndef FPCML
#define FPCML
#include "fpc.hpp"

void fpc_ml(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs);
void fpc2_ml(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs);
unsigned fpc_cpu(ulong* values, unsigned size);

class MLFpc : public IFpc{
    public:
    void fpc(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs) const override{
        return fpc_ml(values,cmp_size_hw,values_size,wgs);
    }
    void fpc2(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs) const override{
        return fpc2_ml(values,cmp_size_hw,values_size,wgs);
    }
};
#endif