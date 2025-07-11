#ifndef Kernel_HPP
#define Kernel_HPP
#include <argparse/argparse.hpp>
#include "Manager.hpp"

class IKernel{
    public:
        virtual ~IKernel() = default;
        virtual int run_kernel(int argc,char** argv) = 0;
    private: 
        virtual void register_cli_options(argparse::ArgumentParser& parser) = 0;
        virtual std::vector<std::string> list_versions() = 0;
};

typedef class_pair<IKernel> kernel_pair ;


struct KernelStats{
    float memcpy2D = 0;
    float memcpy2H = 0;
    float compute = 0;

KernelStats operator+(const KernelStats& other) const {
    return {
        this->memcpy2D + other.memcpy2D,
        this->memcpy2H + other.memcpy2H,
        this->compute + other.compute,
    };
}
KernelStats& operator+=(const KernelStats& other) {
    this->memcpy2D += other.memcpy2D;
    this->memcpy2H += other.memcpy2H;
    this->compute += other.compute;
    return *this;
}
KernelStats& operator/=(float scalar) {
    this->memcpy2D /= scalar;
    this->memcpy2H /= scalar;
    this->compute /= scalar;
    return *this;
}

KernelStats operator/(float scalar) const {
    return {
        this->memcpy2D / scalar,
        this->memcpy2H / scalar,
        this->compute / scalar,
    };
}
};
inline std::ostream& operator<<(std::ostream& os,KernelStats e_stat) {
    os << "Memory load time, to device = " << e_stat.memcpy2D << " ms" << " to Host : "<<  e_stat.memcpy2H << " ms | Compute time =  "<< e_stat.compute << " ms";
    return os;
}



#endif
