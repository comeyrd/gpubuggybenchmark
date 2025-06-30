#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <dispatch/dispatch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <memory>
#include "fpc-reference.hpp"

using namespace MTL;

static NS::SharedPtr<Device> device = NS::TransferPtr(CreateSystemDefaultDevice());

NS::SharedPtr<Library> loadLibrary(const NS::SharedPtr<Device>& device, const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open Metal library: " << path << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read Metal library: " << path << std::endl;
        return {};
    }

    dispatch_data_t dispatchData = dispatch_data_create(
        buffer.data(), buffer.size(),
        dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT
    );

    NS::Error* error = nullptr;
    Library* lib = device->newLibrary(dispatchData, &error);
    if (!lib) {
        std::cerr << "Failed to create library: " << error->localizedDescription()->utf8String() << std::endl;
        return {};
    }

    return NS::TransferPtr(lib);
}

NS::SharedPtr<CommandQueue> getCommandQueue(const NS::SharedPtr<Device>& device) {
    return NS::TransferPtr(device->newCommandQueue());
}

void runKernel(const char* kernelName, const ulong* values, unsigned* cmp_size_hw, int values_size, int wgs) {
    if (values_size % wgs != 0) {
        std::cerr << "values_size must be divisible by wgs\n";
        return;
    }

    static NS::SharedPtr<Library> library = loadLibrary(device, "fpc_kernels.metallib");
    if (!library) return;

    NS::String* nsKernelName = NS::String::string(kernelName, NS::StringEncoding::UTF8StringEncoding);
    Function* kernelFunc = library->newFunction(nsKernelName);
    if (!kernelFunc) {
        std::cerr << "Failed to get function: " << kernelName << std::endl;
        return;
    }

    NS::Error* error = nullptr;
    ComputePipelineState* pipeline = device->newComputePipelineState(kernelFunc, &error);
    kernelFunc->release();
    if (!pipeline) {
        std::cerr << "Failed to create pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        return;
    }

    auto commandQueue = getCommandQueue(device);
    auto valuesBuffer = NS::TransferPtr(device->newBuffer(
        values, values_size * sizeof(ulong), MTL::ResourceStorageModeShared
    ));

    uint32_t zero = 0;
    auto cmpSizeBuffer = NS::TransferPtr(device->newBuffer(
        &zero, sizeof(uint32_t), MTL::ResourceStorageModeShared
    ));

    auto commandBuffer = NS::TransferPtr(commandQueue->commandBuffer());
    auto encoder = NS::TransferPtr(commandBuffer->computeCommandEncoder());

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(valuesBuffer.get(), 0, 0);
    encoder->setBuffer(cmpSizeBuffer.get(), 0, 1);

    MTL::Size gridSize = MTL::Size::Make(values_size, 1, 1);
    MTL::Size threadgroupSize = MTL::Size::Make(wgs, 1, 1);

    encoder->dispatchThreads(gridSize, threadgroupSize);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    std::memcpy(cmp_size_hw, cmpSizeBuffer->contents(), sizeof(uint32_t));
    pipeline->release();
}

void fpc_reference(const ulong* values, unsigned* cmp_size_hw, int values_size, int wgs) {
    runKernel("fpc_reference_kernel", values, cmp_size_hw, values_size, wgs);
}

void fpc2_reference(const ulong* values, unsigned* cmp_size_hw, int values_size, int wgs) {
    runKernel("fpc2_reference_kernel", values, cmp_size_hw, values_size, wgs);
}
REGISTER_CLASS(IFpc,ReferenceFpc);
