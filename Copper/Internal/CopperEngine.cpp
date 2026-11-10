#include "CopperEngine.hpp"

#include "CopperEngineBackend.hpp"
#include "CopperFDTDRunner.h"

#include <cstdio>
#include <exception>

namespace copper {

namespace {
// Backend::GPU (alias Metal) is Metal on macOS and Vulkan elsewhere when Copper was built with it.
// A Vulkan engine that can't start (no driver or device, too little GPU memory) falls back to the
// CPU engine, as a GPU request on a build without any GPU engine always has.
std::unique_ptr<EngineBackend> makeBackend(const CopperYeeGrid& grid, const CopperExcitation& excitation,
                                           const CopperCPML& cpml, CopperEngine::Backend backend,
                                           const CopperDomainMask& domainMask) {
#if defined(__APPLE__) && !defined(COPPER_CPU_ONLY)
    if (backend == CopperEngine::Backend::GPU) return makeMetalEngineBackend(grid, excitation, cpml, domainMask);
#elif defined(COPPER_VULKAN)
    if (backend == CopperEngine::Backend::GPU) {
        try {
            return makeVulkanEngineBackend(grid, excitation, cpml, domainMask);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Copper: Vulkan GPU engine unavailable, using the CPU engine: %s\n", error.what());
        }
    }
#else
    (void)backend;
#endif
    return makeCPUEngineBackend(grid, excitation, cpml, domainMask);
}
} // namespace

CopperEngine::CopperEngine(const CopperYeeGrid& grid, const CopperExcitation& excitation, const CopperCPML& cpml,
                            Backend backend, const CopperDomainMask& domainMask)
    : _backend(makeBackend(grid, excitation, cpml, backend, domainMask)) {}

CopperEngine::~CopperEngine() = default;

void CopperEngine::run(std::uint32_t steps) { _backend->run(steps); }

void CopperEngine::runWithProbeSampling(std::uint32_t steps, const ProbeSampler& sampler,
                                         const MidStepCorrection& midStepCorrection) {
    _backend->runWithProbeSampling(steps, sampler, midStepCorrection);
}

std::vector<float> CopperEngine::readField(Field field) const {
    std::vector<float> result;
    readField(field, result);
    return result;
}

void CopperEngine::readField(Field field, std::vector<float>& destination) const {
    _backend->readField(field, destination);
}

float CopperEngine::readFieldCell(Field field, std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
    return _backend->readFieldCell(field, x, y, z);
}

void CopperEngine::writeFieldCell(Field field, std::uint32_t x, std::uint32_t y, std::uint32_t z, float value) {
    _backend->writeFieldCell(field, x, y, z, value);
}

double CopperEngine::estimateEnergy() const { return _backend->estimateEnergy(); }

const CopperGridDims& CopperEngine::dims() const { return _backend->dims(); }

std::size_t CopperEngine::currentAllocatedMetalBytes() const { return _backend->currentAllocatedMetalBytes(); }

std::size_t CopperEngine::fusedTileCount() const { return _backend->fusedTileCount(); }

void CopperEngine::setLumpedRLC(const std::vector<CopperLumpedRLCCell>& cells) { _backend->setLumpedRLC(cells); }

void CopperEngine::declareMidStepCorrectionCells(const std::vector<CopperLumpedRLCCell>& cells) {
    _backend->declareMidStepCorrectionCells(cells);
}

std::vector<CopperGPUDevice> listGPUDevices(std::string* error) {
#if defined(__APPLE__) && !defined(COPPER_CPU_ONLY)
    // The Metal engine always runs on MTLCreateSystemDefaultDevice().
    (void)error;
    CopperGPUDevice device;
    device.name = "System default GPU";
    device.kind = "other";
    device.api = "Metal";
    device.usable = true;
    device.isDefault = true;
    return {device};
#elif defined(COPPER_VULKAN)
    try {
        return listVulkanDevices();
    } catch (const std::exception& failure) {
        if (error) *error = failure.what();
        return {};
    }
#else
    if (error) *error = "this build of Copper has no GPU engine";
    return {};
#endif
}

} // namespace copper
