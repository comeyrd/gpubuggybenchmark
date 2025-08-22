#ifndef Kernel_HPP
#define Kernel_HPP
#include <argparse/argparse.hpp>
#include "Manager.hpp"
#include <deque>
#include <fstream>

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
    float memcpy2D = 0;//Mem init + copy 2device
    float memcpy2H = 0;//copy back 2 host
    float compute = 0;//Kernel launch time

KernelStats operator+(const KernelStats& other) const {
    return {
        this->memcpy2D + other.memcpy2D,
        this->memcpy2H + other.memcpy2H,
        this->compute + other.compute,
    };
}
KernelStats operator-(const KernelStats& other) const {
    return {
        this->memcpy2D - other.memcpy2D,
        this->memcpy2H - other.memcpy2H,
        this->compute - other.compute,
    };
}
KernelStats& operator-=(const KernelStats& other) {
    this->memcpy2D -= other.memcpy2D;
    this->memcpy2H -= other.memcpy2H;
    this->compute -= other.compute;
    return *this;
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
KernelStats operator/(const KernelStats& other) const{
    return {
        this->memcpy2D / other.memcpy2D,
        this->memcpy2H / other.memcpy2H,
        this->compute / other.compute,
    };
};

KernelStats operator/(float scalar) const {
    return {
        this->memcpy2D / scalar,
        this->memcpy2H / scalar,
        this->compute / scalar,
    };
}
 bool operator<=(uint value) const {
        return std::abs(memcpy2D) <= value && std::abs(memcpy2H) <=value && std::abs(compute) <= value;
    }
};

inline std::ostream& operator<<(std::ostream& os,KernelStats e_stat) {
    os << "Memory load time, to device = " << e_stat.memcpy2D << " ms" << " to Host : "<<  e_stat.memcpy2H << " ms | Compute time =  "<< e_stat.compute << " ms";
    return os;
}
template <typename T>
struct CSVExportable;

template <>
struct CSVExportable<KernelStats> {
    static std::string header() {
        return "memcpy2D,memcpy2H,compute";
    }

    static std::string values(const KernelStats& ks) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << ks.memcpy2D << "," << ks.memcpy2H << "," << ks.compute;
        return oss.str();
    }
};

template <typename Iterable>
void exportCsvToStream(const Iterable& data,std::ostream &os){
    using T = typename Iterable::value_type;
    static_assert(std::is_class_v<CSVExportable<T>>, 
                  "CSVExportable<T> specialization required for this type");
    os << CSVExportable<T>::header() << std::endl;
    for (const auto& item : data){
        os << CSVExportable<T>::values(item) << std::endl;
    }
}

template <typename Iterable>
void exportCsv(const Iterable& data, const std::string &filename){
    using T = typename Iterable::value_type;
    static_assert(std::is_class_v<CSVExportable<T>>, 
                  "CSVExportable<T> specialization required for this type");
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    exportCsvToStream(data,file);
}



class ICriterion{
    public : 
        virtual void observe(const KernelStats& stat) = 0;
        virtual bool should_stop() const = 0;
        virtual ~ICriterion() = default;
};

class ExecutionNumberCriterion: public ICriterion{
    public :
    ExecutionNumberCriterion(int nb_iter):max_iter(nb_iter){}
    void observe(const KernelStats& stat) override{
        count++;
    }
    bool should_stop() const override{
        return count>=max_iter;
    }
    private:
    int max_iter;
    int count = 0;
};

#endif
