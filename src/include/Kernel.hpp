#ifndef Kernel_HPP
#define Kernel_HPP
#include "Manager.hpp"
#include "KernelStats.hpp"
#include <type_traits>
#include <utility>
#include <vector>
#include "gpu-utils.hpp"
#include <argparse/argparse.hpp>
#include "version.hpp"
//Template for Settings
template <typename T>
class has_warmup_and_repetitions {
private:
    template <typename U>
    static auto test(int) -> decltype(std::declval<U>().warmup,std::declval<U>().repetitions,std::is_constructible<U,int,int>::value, std::true_type());

    template <typename U>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

//Template for Data
template <typename Data, typename Settings>
class has_generate_random {
private:
    template <typename U>
    static auto test(int) -> decltype(
        std::declval<U>().generate_random(), std::true_type());

    template <typename U>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<Data>(0))::value;
};


//Template for Data and result -> allocated with Settings info
template <typename Data, typename Settings>
class is_instantiable_by {
private:
    template <typename U>
    static auto test(int) -> decltype(std::is_constructible<U, const Settings&>::value, std::true_type());
    template <typename U>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<Data>(0))::value;
};

template <typename Data, typename Settings, typename Result>
class IVersion{
    public :
    virtual ~IVersion() = default;
    virtual KernelStats run(const Data &data, const Settings &settings,Result &result) const = 0;
};

class I_IKernel {
public:
    virtual ~I_IKernel() = default;
    virtual void run(int argc, char** argv) = 0;
    virtual std::string name() = 0;
    I_IKernel() = default;
};

using kernel_pair = std::pair<const std::string,std::shared_ptr<I_IKernel>>;

constexpr int DEF_WARMUP = 5;
constexpr int DEF_REPETITIONS = 400;

template <typename Data, typename Settings, typename Result>
class IKernel : public I_IKernel {
    static_assert(has_warmup_and_repetitions<Settings>::value,
                  "Settings must have both member variables 'warmup' and 'repetitions'");
    static_assert(has_generate_random<Data,Settings>::value,
                  "Data must have a 'generate_random' function");
    static_assert(is_instantiable_by<Data,Settings>::value,
                  "Data must be instantiable by Settings");
    static_assert(is_instantiable_by<Result,Settings>::value,
                  "Result must be instantiable by Settings");
    protected:
        Settings settings;
        Data data;
        Result cpu_result;

    public: 
        IKernel() : settings(DEF_REPETITIONS,DEF_WARMUP),data(settings),cpu_result(settings){};
        void run(int argc, char** argv) override;
    private:
        virtual void run_cpu() = 0; //PUT THE RESULT OF THE COMPUTATION INSIDE cpu_result !!!
        KernelStats run_impl(std::shared_ptr<IVersion<Data,Settings,Result>> version_impl,Result& result);
        void run_versions(class_umap<IVersion<Data,Settings,Result>> versions);
        void run_versions_repeat_outside(class_umap<IVersion<Data,Settings,Result>> versions,int subrep,int outerrep);
        std::vector<std::string> list_version();
};

#include "Kernel.tpp"

#endif
