#ifndef Kernel_HPP
#define Kernel_HPP
#include "Manager.hpp"
#include "gpu-utils.hpp"
#include "version.hpp"
#include <argparse/argparse.hpp>
#include <type_traits>
#include <utility>
#include <vector>

struct BaseSettings {
    int repetitions;
    int warmup;
    BaseSettings(int _repetitions, int _warmup) : repetitions(_repetitions), warmup(_warmup) {};
};
#include "KernelStats.hpp"

struct BaseData {
    virtual void generate_random() = 0;
    BaseData(const BaseData &) = delete;
    BaseData &operator=(const BaseData &) = delete;
    BaseData(BaseData &&) noexcept = default;
    BaseData &operator=(BaseData &&) noexcept = default;

protected:
    explicit BaseData(const BaseSettings &settings) {}
};

struct BaseResult {
    virtual ~BaseResult() = default;

protected:
    explicit BaseResult(const BaseSettings &settings) {};
};

// Settings must derive from BaseSettings
template <typename T>
struct is_settings_type
    : std::bool_constant<std::is_base_of<BaseSettings, T>::value> {
};

// Data must derive from BaseData
template <typename T>
struct is_data_type
    : std::bool_constant<std::is_base_of<BaseData, T>::value> {};

// Result must derive from BaseResult
template <typename T>
struct is_result_type
    : std::bool_constant<std::is_base_of<BaseResult, T>::value> {};

// Type T must be constructible from const Settings&
template <typename T, typename Settings>
struct is_instantiable_by
    : std::bool_constant<std::is_constructible<T, const Settings &>::value> {};

template <typename Data, typename Settings, typename Result>
class IVersion {
public:
    virtual ~IVersion() = default;
    virtual KernelStats run(const Data &data, const Settings &settings, Result &result) const = 0;
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

template <typename Data, typename Settings, typename Result>
class IKernel : public I_IKernel {
    static_assert(is_settings_type<Settings>::value,
                  "Settings must inherit from BaseSettings");
    static_assert(is_data_type<Data>::value,
                  "Data must inherit from BaseData");
    static_assert(is_result_type<Result>::value,
                  "Result must inherit from BaseResult");
    static_assert(is_instantiable_by<Data, Settings>::value,
                  "Data must be constructible from const Settings&");
    static_assert(is_instantiable_by<Result, Settings>::value,
                  "Result must be constructible from const Settings&");

protected:
    Settings settings;
    Data data;
    Result cpu_result;

public:
    IKernel() : settings(DEF_REPETITIONS, DEF_WARMUP), data(settings), cpu_result(settings) {};
    void run(int argc, char **argv) override;

private:
    virtual void run_cpu() = 0; // PUT THE RESULT OF THE COMPUTATION INSIDE cpu_result !!!
    KernelStats run_impl(std::shared_ptr<IVersion<Data, Settings, Result>> version_impl, Result &result);
    void run_versions(class_umap<IVersion<Data, Settings, Result>> versions);
    void run_versions_repeat_outside(class_umap<IVersion<Data, Settings, Result>> versions, int subrep, int outerrep);
    std::vector<std::string> list_version();
};

#include "Kernel.tpp"

#endif
