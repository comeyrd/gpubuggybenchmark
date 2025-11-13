#ifndef BASETYPESHPP
#define BASETYPESHPP
#include "gpu-utils.hpp"
#include <memory>

struct IData {
    virtual void generate_random() = 0;
    virtual void resize(int work_size) = 0;
    IData(const IData &) = delete;
    IData &operator=(const IData &) = delete;
    IData(IData &&) noexcept = default;
    IData &operator=(IData &&) noexcept = default;
    virtual ~IData() = default;

protected:
    explicit IData(const int &work_size) {}
};

struct IResult {
    virtual ~IResult() = default;
    virtual void resize(int work_size) = 0;

protected:
    explicit IResult(const int &work_size) {};
};

constexpr int DEF_WARMUP = 5;
constexpr int DEF_REPETITIONS = 400;
constexpr int DEF_WORK_SIZE = 1;//Work Size multiplier
constexpr bool DEF_FLUSH_L2_CACHE = true;
constexpr bool DEF_ENABLE_BLOCKING_KERNEL = false;
struct ExecutionConfig{
    int m_repetitions;
    int m_warmups;
    int m_work_size;
    bool m_flush_l2;
    bool m_blocking; 
    ExecutionConfig(int repetitions = DEF_REPETITIONS, int warmups = DEF_WARMUP, int work_size = DEF_WORK_SIZE, bool flush_l2 = DEF_FLUSH_L2_CACHE, bool blocking = DEF_ENABLE_BLOCKING_KERNEL):m_repetitions(repetitions),m_warmups(warmups),m_work_size(work_size),m_flush_l2(flush_l2),m_blocking(blocking){
        if(m_blocking && m_warmups == 0){
            std::cout << "Can't have warmup == 0 and blocking enabled, disabling blocking" << std::endl;
            m_blocking = false;
        }
    };
    static std::vector<ExecutionConfig> generate_all_permutations(std::vector<int> &warmups_v, std::vector<int> &repetitions_v, std::vector<int> &work_size_v, std::vector<bool> &flush_l2_v, std::vector<bool> &blocking_v){
        std::vector<ExecutionConfig> configs;
        configs.reserve(warmups_v.size() * repetitions_v.size() * work_size_v.size() * flush_l2_v.size() * blocking_v.size());
        for (auto w : warmups_v)
            for (auto r : repetitions_v)
                for (auto ws : work_size_v)
                    for (auto f : flush_l2_v)
                        for (auto b : blocking_v)
                            configs.emplace_back(r, w, ws, f, b);
        return configs;
    };
};

#endif