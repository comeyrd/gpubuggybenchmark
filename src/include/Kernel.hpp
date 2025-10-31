#ifndef Kernel_HPP
#define Kernel_HPP
#include "Manager.hpp"
#include "gpu-utils.hpp"
#include "version.hpp"
#include <argparse/argparse.hpp>
#include <type_traits>
#include <utility>
#include <vector>
#include "Types.hpp"
#include "KernelStats.hpp"


// Data must derive from BaseData
template <typename T>
struct is_data_type
    : std::bool_constant<std::is_base_of<IData, T>::value> {};

// Result must derive from BaseResult
template <typename T>
struct is_result_type
    : std::bool_constant<std::is_base_of<IResult, T>::value> {};

// Type T must be constructible from const IData&
template <typename T, typename G>
struct is_instantiable_by
    : std::bool_constant<std::is_constructible<T, const G &>::value> {};

template <typename Data,typename Result>
class IVersion {
public:
    void init(const Data &_data) {
        m_data = &_data;
    }    
    virtual ~IVersion() = default;
    virtual void setup() = 0;
    virtual void reset() = 0;
    virtual void run(stream_t* s) = 0;
    virtual void teardown(Result &_result) = 0;
protected:
    const Data *m_data;
};

class I_IKernel {
public:
    virtual ~I_IKernel() = default;
    virtual void run(int argc, char **argv) = 0;
    virtual std::string name() = 0;
    I_IKernel() = default;
};

using kernel_pair = std::pair<const std::string, std::shared_ptr<I_IKernel>>;

constexpr int DEF_WARMUP = 5;
constexpr int DEF_REPETITIONS = 400;
constexpr int DEF_WORK_SIZE = 1;//Work Size multiplyier
constexpr int DEF_BLOCKING_KERNEL_REP = 10;
template <typename Data, typename Result>
class IKernel : public I_IKernel {
    static_assert(is_data_type<Data>::value,
                  "Data must inherit from IData");
    static_assert(is_result_type<Result>::value,
                  "Result must inherit from BaseResult");
    static_assert(is_instantiable_by<Data, int>::value,
                  "Data must be constructible from const int&");
    static_assert(is_instantiable_by<Result, int>::value,
                  "Result must be constructible from const int&");

protected:
    int m_work_size;
    Data m_data;
    Result m_cpu_result;
    int m_repetitions;
    int m_warmup;
    bool m_flush_l2 = true;//TODO add logic
    bool m_block_kernel = true;
public:
    IKernel() : m_work_size(DEF_WORK_SIZE), m_data(m_work_size), m_cpu_result(m_work_size),m_repetitions(DEF_REPETITIONS),m_warmup(DEF_WARMUP) {};
    void run(int argc, char **argv) override;

private:
    virtual void run_cpu() = 0; // PUT THE RESULT OF THE COMPUTATION INSIDE cpu_result !!!
    KernelStats run_impl(std::shared_ptr<IVersion<Data, Result>> version_impl, Result &result);
    void run_versions(class_umap<IVersion<Data, Result>> versions);
    void run_benchmark(class_umap<IVersion<Data, Result>> versions);
    std::vector<std::string> list_version();
};

#include "Kernel.tpp"

#endif
