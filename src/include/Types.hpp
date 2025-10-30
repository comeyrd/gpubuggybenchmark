#ifndef BASETYPESHPP
#define BASETYPESHPP
#include "gpu-utils.hpp"
#include <memory>

struct BaseSettings {
    int repetitions;
    int warmup;
    BaseSettings(int _repetitions, int _warmup) : repetitions(_repetitions), warmup(_warmup) {};
    ~BaseSettings() = default;

};

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

#endif