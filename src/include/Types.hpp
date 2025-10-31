#ifndef BASETYPESHPP
#define BASETYPESHPP
#include "gpu-utils.hpp"
#include <memory>

struct IData {
    virtual void generate_random() = 0;
    virtual void resize(uint work_size) = 0;
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

protected:
    explicit IResult(const int &work_size) {};
};

#endif