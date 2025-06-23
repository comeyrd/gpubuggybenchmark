#ifndef FPC
#define FPC
#include <unordered_map>
#include <string>
#include <memory>

typedef unsigned long ulong;

class IFpc{
    public: 
        virtual ~IFpc() = default;
        virtual void fpc(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs) const = 0;
        virtual void fpc2(const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs) const = 0;
};

ulong* convertBuffer2Array (char *cbuffer, unsigned size, unsigned step);
void do_fpc(int work_groupe_sz, int repeat);
void run_fpc_impl(std::shared_ptr<IFpc> fpc_impl, ulong* values, unsigned values_size, int cmp_size, int work_group_sz, int repeat);

typedef std::unordered_map<std::string, std::shared_ptr<IFpc>> Kernel_umap;

#endif

