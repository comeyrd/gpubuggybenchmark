#ifndef FPC
#define FPC
typedef unsigned long ulong;

class IFpc{
    public: 
        virtual ~IFpc() = default;
        virtual void fpc(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs) const = 0;
        virtual void fpc2(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs) const = 0;
};

ulong* convertBuffer2Array (char *cbuffer, unsigned size, unsigned step);
void do_fpc(int work_groupe_sz, int repeat);

#endif