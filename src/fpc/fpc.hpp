#ifndef FPC
#define FPC
#include <memory>
#include <string>
#include <unordered_map>

typedef unsigned long ulong;

class IFpc {
public:
    virtual ~IFpc() = default;
    virtual void fpc(const ulong *values, unsigned *cmp_size_hw, const int values_size, const int wgs) const = 0;
    virtual void fpc2(const ulong *values, unsigned *cmp_size_hw, const int values_size, const int wgs) const = 0;
};

ulong *convertBuffer2Array(char *cbuffer, unsigned size, unsigned step);
void do_fpc(int work_groupe_sz, int repeat);
void run_fpc_impl(std::shared_ptr<IFpc> fpc_impl, ulong *values, unsigned values_size, int cmp_size, int work_group_sz, int repeat);

typedef std::unordered_map<std::string, std::shared_ptr<IFpc> > Kernel_umap;


class FPCManager{
    private:
        Kernel_umap _kernels;
    public:
    
        static FPCManager* instance(){
            static FPCManager manager;
            return &manager;
        }
        const std::unordered_map<std::string,std::shared_ptr<IFpc> >&getKernels(){
            return _kernels;
        };

        void registerKernel(const std::string& name, std::shared_ptr<IFpc> kernel){
            _kernels[name] = kernel;
        };
        
};

#define REGISTER_FPC(ClassName) \
    namespace { \
        struct ClassName##AutoRegister { \
            ClassName##AutoRegister() { \
                FPCManager::instance()->registerKernel(#ClassName, std::make_shared<ClassName>()); \
            } \
        }; \
        static ClassName##AutoRegister global_##ClassName##AutoRegister; \
    }

#endif
