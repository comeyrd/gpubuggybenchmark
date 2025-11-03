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

inline const std::string DEF_CSV_PATH = "./csv/all.csv";
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
    Data m_data;
    Result m_cpu_result;
    std::string m_csv_path;
public:
    IKernel() : m_data(DEF_WORK_SIZE), m_cpu_result(DEF_WORK_SIZE) {};
    void run(int argc, char **argv) override;

private:
    virtual void run_cpu() = 0; // PUT THE RESULT OF THE COMPUTATION INSIDE cpu_result !!!
    KernelStats run_impl(std::shared_ptr<IVersion<Data, Result>> version_impl, ExecutionConfig &config, Result &result);
    void run_versions(class_umap<IVersion<Data, Result>> versions,ExecutionConfig &config);
    void run_benchmark(class_umap<IVersion<Data, Result>> versions);
    std::vector<std::string> list_version();
};

#include "Kernel.tpp"

#endif
