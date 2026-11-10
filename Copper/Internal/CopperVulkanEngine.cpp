// Backend::GPU off macOS -- CopperFDTD.metal's plain float kernels as Vulkan compute shaders
// (Copper/Shaders/vulkan), which transcribe CopperCPUEngine's arithmetic cell for cell: E and H
// updates with the CPML folded in, excitation, SERIES lumped RLC, plus a probe gather and an energy
// reduction. It keeps the field layout every Copper backend shares (flat, x fastest,
// copperGridIndex()), one device buffer per array.
//
// Every buffer is reached through its buffer device address, collected in one small host-visible
// Params block (common.glsl) whose own address is the only push constant, so there are no
// descriptor sets at all. The timestep counter lives on the GPU too, so a step's command buffer is
// recorded once and resubmitted for every step.
//
// Probe sampling is what makes a GPU FDTD engine slow if done naively: the runner reads a few dozen
// cells after every step. The first read of a cell copies it back synchronously and adds it to a
// gather list; from then on every step's command buffer ends by copying the listed cells into
// host-visible memory, so later reads cost a hash lookup. writeFieldCell() is deferred until the
// next submission, and its value is served to reads in the meantime.
//
// Built only when CMake finds volk and glslangValidator (COPPER_VULKAN); see CopperEngine.cpp for
// how Backend::GPU picks it and falls back to the CPU engine when no usable device exists.
#if defined(COPPER_VULKAN)

#include "CopperEngineBackend.hpp"
#include "CopperFDTDRunner.h"

#include <volk.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CopperPhysicalConstants.hpp"

// SPIR-V, compiled from Copper/Shaders/vulkan at build time (glslangValidator --vn).
#include "advance.h"
#include "energy.h"
#include "excitation.h"
#include "gather.h"
#include "lumped_rlc.h"
#include "update_e.h"
#include "update_h.h"

namespace copper {

namespace {

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan: ") + what + " failed (VkResult " +
                                 std::to_string(static_cast<int>(result)) + ")");
    }
}

// Mirrors common.glsl's Params member for member (std430: each reference 8 bytes, then the 32-bit
// scalars).
struct VulkanParams {
    VkDeviceAddress e[3], h[3], vv[3], vi[3], ii[3], iv[3];
    VkDeviceAddress layerOf[3], bE[3], cE[3], bH[3], cH[3];
    VkDeviceAddress psi[12];
    VkDeviceAddress state, voltageCells, currentCells, voltageSignal, currentSignal;
    VkDeviceAddress lumpedCells, lumpedState, gatherList, gatherOut, energyPartials;
    std::uint32_t nx, ny, nz;
    std::uint32_t cpmlActive;
    std::uint32_t layerCount[3];
    std::int32_t periodSteps;
    std::uint32_t voltageCount, currentCount, voltageSignalLength, currentSignalLength;
    std::uint32_t lumpedCount, gatherCount;
    std::uint32_t energyCellCount;
};
static_assert(offsetof(VulkanParams, nx) == 55 * sizeof(VkDeviceAddress));
static_assert(offsetof(VulkanParams, energyCellCount) == 55 * sizeof(VkDeviceAddress) + 14 * 4);

// The shaders read these arrays as-is: common.glsl's ExcitationCell and LumpedCell.
static_assert(sizeof(CopperExcitationCell) == 24 && offsetof(CopperExcitationCell, delaySteps) == 20);
static_assert(sizeof(CopperLumpedRLCCell) == 44 && offsetof(CopperLumpedRLCCell, vvd) == 40);

struct PushConstants {
    VkDeviceAddress params;
    std::uint32_t a; // excitation: side; energy: cells per workgroup
    std::uint32_t b; // excitation: serial
};

constexpr std::uint32_t kLocalX = 32, kLocalY = 4, kLocalZ = 2; // the 3D kernels' workgroup size
constexpr std::uint32_t kGatherCapacity = 16384;
constexpr std::uint32_t kStepsPerBatch = 64; // run()'s steps per command buffer
constexpr VkDeviceSize kStagingBytes = VkDeviceSize{64} << 20;

std::uint32_t groups(std::uint32_t count, std::uint32_t local) { return (count + local - 1) / local; }

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// Loads the Vulkan loader and creates a Vulkan 1.2 instance (COPPER_VULKAN_VALIDATION=1 adds the
// Khronos validation layer when it is installed), loading volk's instance functions from it.
VkInstance createInstance() {
    check(volkInitialize(), "loading the Vulkan loader (is a Vulkan driver installed?)");
    if (volkGetInstanceVersion() < VK_API_VERSION_1_2) {
        throw std::runtime_error("Vulkan: the installed loader is older than Vulkan 1.2");
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Copper";
    app.pEngineName = "Copper";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &app;

    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    if (const char* env = std::getenv("COPPER_VULKAN_VALIDATION"); env != nullptr && std::string(env) == "1") {
        std::uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layers(count);
        vkEnumerateInstanceLayerProperties(&count, layers.data());
        const bool available = std::any_of(layers.begin(), layers.end(), [&](const VkLayerProperties& layer) {
            return std::strcmp(layer.layerName, validationLayer) == 0;
        });
        if (available) {
            instanceInfo.enabledLayerCount = 1;
            instanceInfo.ppEnabledLayerNames = &validationLayer;
        } else {
            std::fprintf(stderr, "Copper: COPPER_VULKAN_VALIDATION is set but %s is not installed\n", validationLayer);
        }
    }
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");
    volkLoadInstanceOnly(instance);
    return instance;
}

// One physical device, with what the engine needs to know to choose it.
struct DeviceCandidate {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    std::uint32_t index = 0; // vkEnumeratePhysicalDevices order, which COPPER_VULKAN_DEVICE indexes
    VkPhysicalDeviceProperties properties{};
    std::uint32_t computeFamily = 0;
    std::uint64_t deviceLocalBytes = 0;
    std::string problem; // empty when the engine can run on it
};

std::vector<DeviceCandidate> enumerateDevices(VkInstance instance) {
    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    std::vector<DeviceCandidate> candidates;
    for (std::uint32_t i = 0; i < deviceCount; ++i) {
        DeviceCandidate candidate;
        candidate.device = devices[i];
        candidate.index = i;
        vkGetPhysicalDeviceProperties(devices[i], &candidate.properties);
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &features12;
        if (candidate.properties.apiVersion >= VK_API_VERSION_1_2) vkGetPhysicalDeviceFeatures2(devices[i], &features);

        std::uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &familyCount, families.data());
        candidate.computeFamily = familyCount;
        for (std::uint32_t f = 0; f < familyCount; ++f) {
            if ((families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                candidate.computeFamily = f;
                break;
            }
        }

        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(devices[i], &memory);
        for (std::uint32_t h = 0; h < memory.memoryHeapCount; ++h) {
            if ((memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                candidate.deviceLocalBytes = std::max<std::uint64_t>(candidate.deviceLocalBytes, memory.memoryHeaps[h].size);
            }
        }

        if (candidate.properties.apiVersion < VK_API_VERSION_1_2 || features12.bufferDeviceAddress != VK_TRUE) {
            candidate.problem = "needs Vulkan 1.2 with bufferDeviceAddress";
        } else if (candidate.computeFamily == familyCount) {
            candidate.problem = "has no compute queue";
        }
        candidates.push_back(std::move(candidate));
    }
    return candidates;
}

// How well `candidate` answers a COPPER_VULKAN_DEVICE request (an index, the device's exact name,
// or part of it), or with no request how capable a kind of device it is; negative means "not this
// one". CPU implementations (e.g. llvmpipe), slower than Copper's own CPU engine, are only ever
// picked by request.
int deviceScore(const DeviceCandidate& candidate, const std::string& request) {
    if (!candidate.problem.empty()) return -1;
    const std::string name = candidate.properties.deviceName;
    if (!request.empty()) {
        const bool byIndex =
            std::all_of(request.begin(), request.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
        if (byIndex) return std::stoul(request) == candidate.index ? 300 : -1;
        if (lower(name) == lower(request)) return 200;
        return lower(name).find(lower(request)) != std::string::npos ? 100 : -1;
    }
    switch (candidate.properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
    default: return -1;
    }
}

// The candidate the engine runs on for `request`, or null. Ties go to the first enumerated.
const DeviceCandidate* chooseDevice(const std::vector<DeviceCandidate>& candidates, const std::string& request) {
    const DeviceCandidate* best = nullptr;
    int bestScore = -1;
    for (const DeviceCandidate& candidate : candidates) {
        const int score = deviceScore(candidate, request);
        if (score > bestScore) {
            bestScore = score;
            best = &candidate;
        }
    }
    return best;
}

std::string requestedDevice() {
    const char* wanted = std::getenv("COPPER_VULKAN_DEVICE");
    return wanted != nullptr ? std::string(wanted) : std::string();
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* mapped = nullptr;
};

} // namespace

class VulkanEngineImpl final : public EngineBackend {
public:
    VulkanEngineImpl(const CopperYeeGrid& grid, const CopperExcitation& excitation, const CopperCPML& cpml,
                     const CopperDomainMask& domainMask);
    ~VulkanEngineImpl() override;

    VulkanEngineImpl(const VulkanEngineImpl&) = delete;
    VulkanEngineImpl& operator=(const VulkanEngineImpl&) = delete;

    void run(std::uint32_t steps) override;
    void runWithProbeSampling(std::uint32_t steps, const CopperEngine::ProbeSampler& sampler,
                               const CopperEngine::MidStepCorrection& midStepCorrection) override;
    void readField(CopperEngine::Field field, std::vector<float>& destination) const override;
    float readFieldCell(CopperEngine::Field field, std::uint32_t x, std::uint32_t y, std::uint32_t z) const override;
    void writeFieldCell(CopperEngine::Field field, std::uint32_t x, std::uint32_t y, std::uint32_t z,
                         float value) override;
    double estimateEnergy() const override;
    const CopperGridDims& dims() const override { return _dims; }
    void setLumpedRLC(const std::vector<CopperLumpedRLCCell>& cells) override;
    std::size_t currentAllocatedMetalBytes() const override { return _allocatedBytes; }

private:
    // --- device ---
    void createDevice();
    void release();
    std::uint32_t memoryType(std::uint32_t typeBits, VkMemoryPropertyFlags required,
                             VkMemoryPropertyFlags preferred) const;
    Buffer createBuffer(VkDeviceSize bytes, bool hostVisible, bool hostCached = false);
    void destroyBuffer(Buffer& buffer);
    VkCommandBuffer allocateCommandBuffer();
    VkPipeline createPipeline(const std::uint32_t* code, std::size_t bytes);

    // --- transfers (each submits and waits) ---
    void zeroBuffers(const std::vector<Buffer*>& buffers);
    void upload(const Buffer& destination, const void* data, VkDeviceSize bytes);
    void download(const Buffer& source, VkDeviceSize offset, void* data, VkDeviceSize bytes) const;

    // --- recording ---
    void dispatch(VkCommandBuffer cb, VkPipeline pipeline, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                  std::uint32_t a = 0, std::uint32_t b = 0) const;
    static void barrier(VkCommandBuffer cb);
    static void hostBarrier(VkCommandBuffer cb);
    void recordVoltage(VkCommandBuffer cb) const;
    void recordCurrent(VkCommandBuffer cb) const;
    void recordGather(VkCommandBuffer cb) const;
    void begin(VkCommandBuffer cb, bool oneTime) const;
    void end(VkCommandBuffer cb) const;
    void ensureRecorded();

    /// Submits `cb` after any deferred writeFieldCell() values and waits for it.
    void submit(VkCommandBuffer cb) const;

    const Buffer& fieldBuffer(int component) const { return component < 3 ? _e[component] : _h[component - 3]; }
    static std::uint64_t cellKey(int component, std::uint32_t index) {
        return (static_cast<std::uint64_t>(component) << 32) | index;
    }

    // Vulkan objects.
    VkInstance _instance = VK_NULL_HANDLE;
    VkPhysicalDevice _physical = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties _memoryProperties{};
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;
    std::uint32_t _queueFamily = 0;
    VkCommandPool _pool = VK_NULL_HANDLE;
    VkFence _fence = VK_NULL_HANDLE;
    VkPipelineLayout _layout = VK_NULL_HANDLE;
    VkPipeline _updateE = VK_NULL_HANDLE, _updateH = VK_NULL_HANDLE, _excitation = VK_NULL_HANDLE,
               _lumped = VK_NULL_HANDLE, _advance = VK_NULL_HANDLE, _gather = VK_NULL_HANDLE,
               _energy = VK_NULL_HANDLE;
    VkCommandBuffer _stepCB = VK_NULL_HANDLE, _voltageCB = VK_NULL_HANDLE, _currentCB = VK_NULL_HANDLE,
                    _batchCB = VK_NULL_HANDLE, _writeCB = VK_NULL_HANDLE, _scratchCB = VK_NULL_HANDLE;
    bool _recorded = false;
    std::size_t _allocatedBytes = 0;

    // Buffers.
    Buffer _e[3], _h[3], _vv[3], _vi[3], _ii[3], _iv[3];
    Buffer _layerOf[3], _bE[3], _cE[3], _bH[3], _cH[3];
    Buffer _psi[12];
    Buffer _state, _voltageCells, _currentCells, _voltageSignal, _currentSignal;
    Buffer _lumpedCells, _lumpedState;
    Buffer _gatherList, _gatherOut, _energyPartials;
    Buffer _params;
    mutable Buffer _staging;
    mutable Buffer _cellReadback;

    CopperGridDims _dims;
    std::uint32_t _currentTimestep = 0;
    bool _serialVoltage = false, _serialCurrent = false;
    std::uint32_t _energyCellsPerGroup = 0, _energyGroups = 0;

    // Probe cells copied back after every submission: key -> slot in _gatherOut.
    mutable std::unordered_map<std::uint64_t, std::uint32_t> _gatherSlots;
    mutable bool _gatherFullReported = false;
    // writeFieldCell() values not yet on the device.
    mutable std::unordered_map<std::uint64_t, float> _pendingWrites;

    VulkanParams* params() const { return static_cast<VulkanParams*>(_params.mapped); }
};

// ---------------------------------------------------------------------------------------------
// Device setup

void VulkanEngineImpl::createDevice() {
    _instance = createInstance();

    // COPPER_VULKAN_DEVICE picks a device by index or by (part of) its name; otherwise the most
    // capable kind present (see deviceScore()).
    const std::vector<DeviceCandidate> candidates = enumerateDevices(_instance);
    const std::string request = requestedDevice();
    const DeviceCandidate* chosen = chooseDevice(candidates, request);
    if (chosen == nullptr) {
        std::string message = "Vulkan: no usable GPU";
        if (!request.empty()) message += " matching COPPER_VULKAN_DEVICE=" + request;
        std::string rejected;
        for (const DeviceCandidate& candidate : candidates) {
            if (!candidate.problem.empty()) {
                rejected += std::string(" ") + candidate.properties.deviceName + " (" + candidate.problem + ");";
            }
        }
        if (!rejected.empty()) message += " --" + rejected;
        throw std::runtime_error(message);
    }
    _physical = chosen->device;
    _queueFamily = chosen->computeFamily;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(_physical, &properties);
    vkGetPhysicalDeviceMemoryProperties(_physical, &_memoryProperties);
    std::fprintf(stderr, "Copper: Vulkan device: %s (Vulkan %u.%u)\n", properties.deviceName,
                 VK_API_VERSION_MAJOR(properties.apiVersion), VK_API_VERSION_MINOR(properties.apiVersion));

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = _queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enable12.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &enable12;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    check(vkCreateDevice(_physical, &deviceInfo, nullptr, &_device), "vkCreateDevice");
    volkLoadDevice(_device);
    vkGetDeviceQueue(_device, _queueFamily, 0, &_queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = _queueFamily;
    check(vkCreateCommandPool(_device, &poolInfo, nullptr, &_pool), "vkCreateCommandPool");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(_device, &fenceInfo, nullptr, &_fence), "vkCreateFence");

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    check(vkCreatePipelineLayout(_device, &layoutInfo, nullptr, &_layout), "vkCreatePipelineLayout");

    _updateE = createPipeline(copper_vk_update_e, sizeof(copper_vk_update_e));
    _updateH = createPipeline(copper_vk_update_h, sizeof(copper_vk_update_h));
    _excitation = createPipeline(copper_vk_excitation, sizeof(copper_vk_excitation));
    _lumped = createPipeline(copper_vk_lumped_rlc, sizeof(copper_vk_lumped_rlc));
    _advance = createPipeline(copper_vk_advance, sizeof(copper_vk_advance));
    _gather = createPipeline(copper_vk_gather, sizeof(copper_vk_gather));
    _energy = createPipeline(copper_vk_energy, sizeof(copper_vk_energy));

    _stepCB = allocateCommandBuffer();
    _voltageCB = allocateCommandBuffer();
    _currentCB = allocateCommandBuffer();
    _batchCB = allocateCommandBuffer();
    _writeCB = allocateCommandBuffer();
    _scratchCB = allocateCommandBuffer();
}

VkPipeline VulkanEngineImpl::createPipeline(const std::uint32_t* code, std::size_t bytes) {
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = bytes;
    moduleInfo.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(_device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = _layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(_device, module, nullptr);
    check(result, "vkCreateComputePipelines");
    return pipeline;
}

VkCommandBuffer VulkanEngineImpl::allocateCommandBuffer() {
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool = _pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(_device, &info, &cb), "vkAllocateCommandBuffers");
    return cb;
}

std::uint32_t VulkanEngineImpl::memoryType(std::uint32_t typeBits, VkMemoryPropertyFlags required,
                                           VkMemoryPropertyFlags preferred) const {
    std::uint32_t fallback = UINT32_MAX;
    for (std::uint32_t i = 0; i < _memoryProperties.memoryTypeCount; ++i) {
        if ((typeBits & (1U << i)) == 0) continue;
        const VkMemoryPropertyFlags flags = _memoryProperties.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if ((flags & preferred) == preferred) return i;
        if (fallback == UINT32_MAX) fallback = i;
    }
    if (fallback == UINT32_MAX) throw std::runtime_error("Vulkan: no suitable memory type");
    return fallback;
}

Buffer VulkanEngineImpl::createBuffer(VkDeviceSize bytes, bool hostVisible, bool hostCached) {
    Buffer buffer;
    buffer.size = std::max<VkDeviceSize>(bytes, 16); // empty arrays still get a valid address
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = buffer.size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(_device, &info, nullptr, &buffer.buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(_device, buffer.buffer, &requirements);
    const VkMemoryPropertyFlags required =
        hostVisible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkMemoryPropertyFlags preferred = hostCached ? required | VK_MEMORY_PROPERTY_HOST_CACHED_BIT : required;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.pNext = &flags;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType(requirements.memoryTypeBits, required, preferred);
    const VkResult result = vkAllocateMemory(_device, &allocation, nullptr, &buffer.memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(_device, buffer.buffer, nullptr);
        throw std::runtime_error("Vulkan: could not allocate " + std::to_string(requirements.size >> 20) +
                                 " MiB of " + (hostVisible ? "host-visible" : "GPU") + " memory (VkResult " +
                                 std::to_string(static_cast<int>(result)) + ", " +
                                 std::to_string(_allocatedBytes >> 20) + " MiB already allocated)");
    }
    _allocatedBytes += requirements.size;
    check(vkBindBufferMemory(_device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory");
    if (hostVisible) check(vkMapMemory(_device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped), "vkMapMemory");

    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer.buffer;
    buffer.address = vkGetBufferDeviceAddress(_device, &addressInfo);
    return buffer;
}

void VulkanEngineImpl::destroyBuffer(Buffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(_device, buffer.buffer, nullptr);
    if (buffer.memory != VK_NULL_HANDLE) vkFreeMemory(_device, buffer.memory, nullptr);
    buffer = {};
}

// ---------------------------------------------------------------------------------------------
// Construction

VulkanEngineImpl::VulkanEngineImpl(const CopperYeeGrid& grid, const CopperExcitation& excitation,
                                   const CopperCPML& cpml, const CopperDomainMask& domainMask)
    : _dims(grid.dims) {
    try {
        createDevice();

        const std::size_t cellCount = _dims.cellCount();
        const VkDeviceSize fieldBytes = cellCount * sizeof(float);
        std::vector<Buffer*> zeroed;
        for (int axis = 0; axis < 3; ++axis) {
            _e[axis] = createBuffer(fieldBytes, false);
            _h[axis] = createBuffer(fieldBytes, false);
            zeroed.push_back(&_e[axis]);
            zeroed.push_back(&_h[axis]);
        }

        // Coefficients: the domain mask's external nodes are zeroed and the ring absorber folded in
        // on host copies, exactly as CopperCPUEngine does, then each array is uploaded.
        {
            std::vector<float> vv[3], vi[3], ii[3], iv[3];
            for (int axis = 0; axis < 3; ++axis) {
                vv[axis] = grid.vv[axis];
                vi[axis] = grid.vi[axis];
                ii[axis] = grid.ii[axis];
                iv[axis] = grid.iv[axis];
                if (!domainMask.empty()) {
                    for (std::uint32_t y = 0; y < _dims.ny; ++y) {
                        for (std::uint32_t x = 0; x < _dims.nx; ++x) {
                            if (domainMask.at(x, y) != 0) continue;
                            for (std::uint32_t z = 0; z < _dims.nz; ++z) {
                                const std::size_t i = copperGridIndex(_dims, x, y, z);
                                vv[axis][i] = vi[axis][i] = ii[axis][i] = iv[axis][i] = 0.0F;
                            }
                        }
                    }
                }
            }
            float* const pvv[3] = {vv[0].data(), vv[1].data(), vv[2].data()};
            float* const pvi[3] = {vi[0].data(), vi[1].data(), vi[2].data()};
            float* const pii[3] = {ii[0].data(), ii[1].data(), ii[2].data()};
            float* const piv[3] = {iv[0].data(), iv[1].data(), iv[2].data()};
            applyRingAbsorber(domainMask, grid.timestepSeconds, _dims, pvv, pvi, pii, piv);

            _staging = createBuffer(std::min<VkDeviceSize>(kStagingBytes, std::max<VkDeviceSize>(fieldBytes, 1024)),
                                    true, true);
            for (int axis = 0; axis < 3; ++axis) {
                for (auto [buffer, values] : {std::pair{&_vv[axis], &vv[axis]}, std::pair{&_vi[axis], &vi[axis]},
                                              std::pair{&_ii[axis], &ii[axis]}, std::pair{&_iv[axis], &iv[axis]}}) {
                    if (values->size() != cellCount) {
                        throw std::runtime_error("Vulkan engine: a coefficient array does not match the grid");
                    }
                    *buffer = createBuffer(fieldBytes, false);
                    upload(*buffer, values->data(), fieldBytes);
                    std::vector<float>().swap(*values); // keep the host peak at one copy
                }
            }
        }

        // CPML.
        const std::uint32_t lines[3] = {_dims.nx, _dims.ny, _dims.nz};
        const bool cpmlActive = !cpml.empty();
        for (int axis = 0; axis < 3; ++axis) {
            const CopperCPML::Axis& graded = cpml.axes[axis];
            std::vector<std::uint32_t> layerOf = graded.layerOf;
            layerOf.resize(lines[axis], CopperCPML::kNoLayer);
            _layerOf[axis] = createBuffer(layerOf.size() * sizeof(std::uint32_t), false);
            upload(_layerOf[axis], layerOf.data(), layerOf.size() * sizeof(std::uint32_t));
            Buffer* const perLayer[4] = {&_bE[axis], &_cE[axis], &_bH[axis], &_cH[axis]};
            const std::vector<float>* const values[4] = {&graded.bE, &graded.cE, &graded.bH, &graded.cH};
            for (int k = 0; k < 4; ++k) {
                *perLayer[k] = createBuffer(values[k]->size() * sizeof(float), false);
                zeroed.push_back(perLayer[k]);
            }
            const std::size_t psiCount = cpmlActive ? cpml.psiCount(_dims, axis) : 0;
            for (int side = 0; side < 2; ++side) {
                for (int term = 0; term < 2; ++term) {
                    Buffer& psi = _psi[side * 6 + axis * 2 + term];
                    psi = createBuffer(psiCount * sizeof(float), false);
                    zeroed.push_back(&psi);
                }
            }
        }

        // Excitation.
        auto excitationCells = [&](Buffer& buffer, const std::vector<CopperExcitationCell>& cells) {
            buffer = createBuffer(cells.size() * sizeof(CopperExcitationCell), false);
            zeroed.push_back(&buffer);
        };
        excitationCells(_voltageCells, excitation.voltageCells);
        excitationCells(_currentCells, excitation.currentCells);
        _voltageSignal = createBuffer(excitation.voltageSignal.size() * sizeof(float), false);
        _currentSignal = createBuffer(excitation.currentSignal.size() * sizeof(float), false);
        zeroed.push_back(&_voltageSignal);
        zeroed.push_back(&_currentSignal);
        auto hasSharedField = [](const std::vector<CopperExcitationCell>& cells) {
            std::unordered_set<std::uint64_t> seen;
            for (const CopperExcitationCell& cell : cells) {
                const std::uint64_t key = (static_cast<std::uint64_t>(cell.axis) << 62) ^
                                          (static_cast<std::uint64_t>(cell.z) << 42) ^
                                          (static_cast<std::uint64_t>(cell.y) << 21) ^ cell.x;
                if (!seen.insert(key).second) return true;
            }
            return false;
        };
        _serialVoltage = hasSharedField(excitation.voltageCells);
        _serialCurrent = hasSharedField(excitation.currentCells);

        _state = createBuffer(16, false);
        zeroed.push_back(&_state);
        _lumpedCells = createBuffer(0, false);
        _lumpedState = createBuffer(0, false);
        zeroed.push_back(&_lumpedCells);
        zeroed.push_back(&_lumpedState);

        _gatherList = createBuffer(kGatherCapacity * 2 * sizeof(std::uint32_t), true);
        _gatherOut = createBuffer(kGatherCapacity * sizeof(float), true, true);
        _cellReadback = createBuffer(16, true, true);

        // estimateEnergy()'s reduction: >= 1024 cells per workgroup, and few enough workgroups for
        // one dispatch dimension.
        _energyCellsPerGroup = std::max<std::uint32_t>(1024, groups(_dims.cellCount(), 65535));
        _energyCellsPerGroup = groups(_energyCellsPerGroup, 256) * 256;
        _energyGroups = std::max<std::uint32_t>(1, groups(_dims.cellCount(), _energyCellsPerGroup));
        _energyPartials = createBuffer(VkDeviceSize{_energyGroups} * 2 * sizeof(float), true, true);

        zeroBuffers(zeroed);
        for (int axis = 0; axis < 3; ++axis) {
            const CopperCPML::Axis& graded = cpml.axes[axis];
            upload(_bE[axis], graded.bE.data(), graded.bE.size() * sizeof(float));
            upload(_cE[axis], graded.cE.data(), graded.cE.size() * sizeof(float));
            upload(_bH[axis], graded.bH.data(), graded.bH.size() * sizeof(float));
            upload(_cH[axis], graded.cH.data(), graded.cH.size() * sizeof(float));
        }
        upload(_voltageCells, excitation.voltageCells.data(),
               excitation.voltageCells.size() * sizeof(CopperExcitationCell));
        upload(_currentCells, excitation.currentCells.data(),
               excitation.currentCells.size() * sizeof(CopperExcitationCell));
        upload(_voltageSignal, excitation.voltageSignal.data(), excitation.voltageSignal.size() * sizeof(float));
        upload(_currentSignal, excitation.currentSignal.data(), excitation.currentSignal.size() * sizeof(float));

        _params = createBuffer(sizeof(VulkanParams), true);
        VulkanParams& p = *params();
        p = {};
        for (int axis = 0; axis < 3; ++axis) {
            p.e[axis] = _e[axis].address;
            p.h[axis] = _h[axis].address;
            p.vv[axis] = _vv[axis].address;
            p.vi[axis] = _vi[axis].address;
            p.ii[axis] = _ii[axis].address;
            p.iv[axis] = _iv[axis].address;
            p.layerOf[axis] = _layerOf[axis].address;
            p.bE[axis] = _bE[axis].address;
            p.cE[axis] = _cE[axis].address;
            p.bH[axis] = _bH[axis].address;
            p.cH[axis] = _cH[axis].address;
            p.layerCount[axis] = cpml.axes[axis].layerCount();
        }
        for (int i = 0; i < 12; ++i) p.psi[i] = _psi[i].address;
        p.state = _state.address;
        p.voltageCells = _voltageCells.address;
        p.currentCells = _currentCells.address;
        p.voltageSignal = _voltageSignal.address;
        p.currentSignal = _currentSignal.address;
        p.lumpedCells = _lumpedCells.address;
        p.lumpedState = _lumpedState.address;
        p.gatherList = _gatherList.address;
        p.gatherOut = _gatherOut.address;
        p.energyPartials = _energyPartials.address;
        p.nx = _dims.nx;
        p.ny = _dims.ny;
        p.nz = _dims.nz;
        p.cpmlActive = cpmlActive ? 1U : 0U;
        // CopperCPUEngine's period: the signal's own when it is periodic, else none (numTS + 1).
        if (excitation.signalPeriodSeconds > 0.0) {
            p.periodSteps = std::max<std::int32_t>(
                1, static_cast<std::int32_t>(excitation.signalPeriodSeconds / grid.timestepSeconds));
        }
        p.voltageCount = static_cast<std::uint32_t>(excitation.voltageCells.size());
        p.currentCount = static_cast<std::uint32_t>(excitation.currentCells.size());
        p.voltageSignalLength = static_cast<std::uint32_t>(excitation.voltageSignal.size());
        p.currentSignalLength = static_cast<std::uint32_t>(excitation.currentSignal.size());
        p.energyCellCount = _dims.cellCount();
    } catch (...) {
        release(); // a throwing constructor skips the destructor
        throw;
    }
}

VulkanEngineImpl::~VulkanEngineImpl() { release(); }

void VulkanEngineImpl::release() {
    if (_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(_device);
        for (Buffer* group : {_e, _h, _vv, _vi, _ii, _iv, _layerOf, _bE, _cE, _bH, _cH}) {
            for (int i = 0; i < 3; ++i) destroyBuffer(group[i]);
        }
        for (Buffer& psi : _psi) destroyBuffer(psi);
        for (Buffer* buffer : {&_state, &_voltageCells, &_currentCells, &_voltageSignal, &_currentSignal,
                               &_lumpedCells, &_lumpedState, &_gatherList, &_gatherOut, &_energyPartials, &_params,
                               &_staging, &_cellReadback}) {
            destroyBuffer(*buffer);
        }
        for (VkPipeline* pipeline : {&_updateE, &_updateH, &_excitation, &_lumped, &_advance, &_gather, &_energy}) {
            if (*pipeline != VK_NULL_HANDLE) vkDestroyPipeline(_device, *pipeline, nullptr);
            *pipeline = VK_NULL_HANDLE;
        }
        if (_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(_device, _layout, nullptr);
        if (_fence != VK_NULL_HANDLE) vkDestroyFence(_device, _fence, nullptr);
        if (_pool != VK_NULL_HANDLE) vkDestroyCommandPool(_device, _pool, nullptr);
        vkDestroyDevice(_device, nullptr);
        _device = VK_NULL_HANDLE;
    }
    if (_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(_instance, nullptr);
        _instance = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------------------------
// Transfers

void VulkanEngineImpl::begin(VkCommandBuffer cb, bool oneTime) const {
    check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = oneTime ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0;
    check(vkBeginCommandBuffer(cb, &info), "vkBeginCommandBuffer");
}

void VulkanEngineImpl::end(VkCommandBuffer cb) const { check(vkEndCommandBuffer(cb), "vkEndCommandBuffer"); }

void VulkanEngineImpl::submit(VkCommandBuffer cb) const {
    VkCommandBuffer buffers[2];
    std::uint32_t count = 0;
    if (!_pendingWrites.empty()) {
        begin(_writeCB, true);
        for (const auto& [key, value] : _pendingWrites) {
            const int component = static_cast<int>(key >> 32);
            const VkDeviceSize offset = (key & 0xFFFFFFFFULL) * sizeof(float);
            vkCmdUpdateBuffer(_writeCB, fieldBuffer(component).buffer, offset, sizeof(float), &value);
        }
        barrier(_writeCB);
        end(_writeCB);
        buffers[count++] = _writeCB;
    }
    buffers[count++] = cb;
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = count;
    info.pCommandBuffers = buffers;
    check(vkResetFences(_device, 1, &_fence), "vkResetFences");
    check(vkQueueSubmit(_queue, 1, &info, _fence), "vkQueueSubmit");
    check(vkWaitForFences(_device, 1, &_fence, VK_TRUE, UINT64_MAX), "waiting for the GPU");
    _pendingWrites.clear();
}

void VulkanEngineImpl::zeroBuffers(const std::vector<Buffer*>& buffers) {
    begin(_scratchCB, true);
    for (const Buffer* buffer : buffers) vkCmdFillBuffer(_scratchCB, buffer->buffer, 0, VK_WHOLE_SIZE, 0);
    barrier(_scratchCB);
    end(_scratchCB);
    submit(_scratchCB);
}

void VulkanEngineImpl::upload(const Buffer& destination, const void* data, VkDeviceSize bytes) {
    const auto* source = static_cast<const std::uint8_t*>(data);
    for (VkDeviceSize offset = 0; offset < bytes; offset += _staging.size) {
        const VkDeviceSize chunk = std::min(_staging.size, bytes - offset);
        std::memcpy(_staging.mapped, source + offset, chunk);
        begin(_scratchCB, true);
        const VkBufferCopy region{0, offset, chunk};
        vkCmdCopyBuffer(_scratchCB, _staging.buffer, destination.buffer, 1, &region);
        barrier(_scratchCB);
        end(_scratchCB);
        submit(_scratchCB);
    }
}

void VulkanEngineImpl::download(const Buffer& source, VkDeviceSize offset, void* data, VkDeviceSize bytes) const {
    auto* target = static_cast<std::uint8_t*>(data);
    for (VkDeviceSize done = 0; done < bytes; done += _staging.size) {
        const VkDeviceSize chunk = std::min(_staging.size, bytes - done);
        begin(_scratchCB, true);
        const VkBufferCopy region{offset + done, 0, chunk};
        vkCmdCopyBuffer(_scratchCB, source.buffer, _staging.buffer, 1, &region);
        hostBarrier(_scratchCB);
        end(_scratchCB);
        submit(_scratchCB);
        std::memcpy(target + done, _staging.mapped, chunk);
    }
}

// ---------------------------------------------------------------------------------------------
// Recording

void VulkanEngineImpl::barrier(VkCommandBuffer cb) {
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cb, stages, stages, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

void VulkanEngineImpl::hostBarrier(VkCommandBuffer cb) {
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

void VulkanEngineImpl::dispatch(VkCommandBuffer cb, VkPipeline pipeline, std::uint32_t gx, std::uint32_t gy,
                                std::uint32_t gz, std::uint32_t a, std::uint32_t b) const {
    if (gx == 0 || gy == 0 || gz == 0) return;
    const PushConstants push{_params.address, a, b};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cb, _layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cb, gx, gy, gz);
    barrier(cb);
}

// The voltage half-step, in CopperCPUEngine::runIterationPhase's order: E update with its CPML,
// voltage excitation, lumped RLC.
void VulkanEngineImpl::recordVoltage(VkCommandBuffer cb) const {
    const VulkanParams& p = *params();
    dispatch(cb, _updateE, groups(_dims.nx, kLocalX), groups(_dims.ny, kLocalY), groups(_dims.nz, kLocalZ));
    if (p.voltageCount != 0) dispatch(cb, _excitation, groups(p.voltageCount, 64), 1, 1, 0, _serialVoltage ? 1 : 0);
    if (p.lumpedCount != 0) dispatch(cb, _lumped, groups(p.lumpedCount, 64), 1, 1);
}

// The current half-step: H update with its CPML, current excitation, then the timestep advances.
void VulkanEngineImpl::recordCurrent(VkCommandBuffer cb) const {
    const VulkanParams& p = *params();
    if (_dims.nx > 1 && _dims.ny > 1 && _dims.nz > 1) {
        dispatch(cb, _updateH, groups(_dims.nx - 1, kLocalX), groups(_dims.ny - 1, kLocalY),
                 groups(_dims.nz - 1, kLocalZ));
    }
    if (p.currentCount != 0) dispatch(cb, _excitation, groups(p.currentCount, 64), 1, 1, 1, _serialCurrent ? 1 : 0);
    dispatch(cb, _advance, 1, 1, 1);
}

// Sized for the whole gather list's capacity: cells join it between submissions, and the shader
// stops at the current count.
void VulkanEngineImpl::recordGather(VkCommandBuffer cb) const {
    dispatch(cb, _gather, groups(kGatherCapacity, 64), 1, 1);
    hostBarrier(cb);
}

void VulkanEngineImpl::ensureRecorded() {
    if (_recorded) return;
    begin(_stepCB, false);
    recordVoltage(_stepCB);
    recordCurrent(_stepCB);
    recordGather(_stepCB);
    end(_stepCB);

    begin(_voltageCB, false);
    recordVoltage(_voltageCB);
    recordGather(_voltageCB);
    end(_voltageCB);

    begin(_currentCB, false);
    recordCurrent(_currentCB);
    recordGather(_currentCB);
    end(_currentCB);

    begin(_batchCB, false);
    for (std::uint32_t step = 0; step < kStepsPerBatch; ++step) {
        recordVoltage(_batchCB);
        recordCurrent(_batchCB);
    }
    recordGather(_batchCB);
    end(_batchCB);
    _recorded = true;
}

// ---------------------------------------------------------------------------------------------
// EngineBackend

void VulkanEngineImpl::run(std::uint32_t steps) {
    ensureRecorded();
    std::uint32_t done = 0;
    for (; done + kStepsPerBatch <= steps; done += kStepsPerBatch) submit(_batchCB);
    for (; done < steps; ++done) submit(_stepCB);
    _currentTimestep += steps;
}

void VulkanEngineImpl::runWithProbeSampling(std::uint32_t steps, const CopperEngine::ProbeSampler& sampler,
                                             const CopperEngine::MidStepCorrection& midStepCorrection) {
    ensureRecorded();
    for (std::uint32_t step = 0; step < steps; ++step) {
        if (midStepCorrection) {
            submit(_voltageCB);
            midStepCorrection();
            submit(_currentCB);
        } else {
            submit(_stepCB);
        }
        ++_currentTimestep;
        if (!sampler(_currentTimestep)) break;
    }
}

void VulkanEngineImpl::readField(CopperEngine::Field field, std::vector<float>& destination) const {
    const int component = static_cast<int>(field);
    destination.resize(_dims.cellCount());
    download(fieldBuffer(component), 0, destination.data(), destination.size() * sizeof(float));
}

float VulkanEngineImpl::readFieldCell(CopperEngine::Field field, std::uint32_t x, std::uint32_t y,
                                      std::uint32_t z) const {
    const int component = static_cast<int>(field);
    const std::uint32_t index = copperGridIndex(_dims, x, y, z);
    const std::uint64_t key = cellKey(component, index);
    if (const auto pending = _pendingWrites.find(key); pending != _pendingWrites.end()) return pending->second;
    if (const auto slot = _gatherSlots.find(key); slot != _gatherSlots.end()) {
        return static_cast<const float*>(_gatherOut.mapped)[slot->second];
    }

    // First read of this cell: copy it back now, and gather it after every submission from here on.
    begin(_scratchCB, true);
    const VkBufferCopy region{VkDeviceSize{index} * sizeof(float), 0, sizeof(float)};
    vkCmdCopyBuffer(_scratchCB, fieldBuffer(component).buffer, _cellReadback.buffer, 1, &region);
    hostBarrier(_scratchCB);
    end(_scratchCB);
    submit(_scratchCB);
    const float value = *static_cast<const float*>(_cellReadback.mapped);

    VulkanParams& p = *params();
    if (p.gatherCount < kGatherCapacity) {
        const std::uint32_t slot = p.gatherCount;
        auto* list = static_cast<std::uint32_t*>(_gatherList.mapped);
        list[2 * slot] = static_cast<std::uint32_t>(component);
        list[2 * slot + 1] = index;
        static_cast<float*>(_gatherOut.mapped)[slot] = value;
        _gatherSlots.emplace(key, slot);
        p.gatherCount = slot + 1;
    } else if (!_gatherFullReported) {
        _gatherFullReported = true;
        std::fprintf(stderr, "Copper: Vulkan probe gather list is full (%u cells); further cells read slowly\n",
                     kGatherCapacity);
    }
    return value;
}

void VulkanEngineImpl::writeFieldCell(CopperEngine::Field field, std::uint32_t x, std::uint32_t y, std::uint32_t z,
                                      float value) {
    const int component = static_cast<int>(field);
    const std::uint32_t index = copperGridIndex(_dims, x, y, z);
    const std::uint64_t key = cellKey(component, index);
    _pendingWrites[key] = value;
    if (const auto slot = _gatherSlots.find(key); slot != _gatherSlots.end()) {
        static_cast<float*>(_gatherOut.mapped)[slot->second] = value;
    }
}

double VulkanEngineImpl::estimateEnergy() const {
    begin(_scratchCB, true);
    dispatch(_scratchCB, _energy, _energyGroups, 1, 1, _energyCellsPerGroup);
    hostBarrier(_scratchCB);
    end(_scratchCB);
    submit(_scratchCB);
    const auto* partials = static_cast<const float*>(_energyPartials.mapped);
    double eSumSq = 0.0, hSumSq = 0.0;
    for (std::uint32_t g = 0; g < _energyGroups; ++g) {
        eSumSq += static_cast<double>(partials[2 * g]);
        hSumSq += static_cast<double>(partials[2 * g + 1]);
    }
    return physical::epsilon0 * eSumSq + physical::mu0 * hSumSq;
}

void VulkanEngineImpl::setLumpedRLC(const std::vector<CopperLumpedRLCCell>& cells) {
    destroyBuffer(_lumpedCells);
    destroyBuffer(_lumpedState);
    _lumpedCells = createBuffer(cells.size() * sizeof(CopperLumpedRLCCell), false);
    _lumpedState = createBuffer(cells.size() * 6 * sizeof(float), false);
    zeroBuffers({&_lumpedCells, &_lumpedState});
    upload(_lumpedCells, cells.data(), cells.size() * sizeof(CopperLumpedRLCCell));
    VulkanParams& p = *params();
    p.lumpedCells = _lumpedCells.address;
    p.lumpedState = _lumpedState.address;
    p.lumpedCount = static_cast<std::uint32_t>(cells.size());
    _recorded = false; // the voltage half-step gains or loses its lumped dispatch
}

std::unique_ptr<EngineBackend> makeVulkanEngineBackend(const CopperYeeGrid& grid, const CopperExcitation& excitation,
                                                       const CopperCPML& cpml, const CopperDomainMask& domainMask) {
    return std::make_unique<VulkanEngineImpl>(grid, excitation, cpml, domainMask);
}

std::vector<CopperGPUDevice> listVulkanDevices() {
    const VkInstance instance = createInstance();
    std::vector<CopperGPUDevice> result;
    try {
        const std::vector<DeviceCandidate> candidates = enumerateDevices(instance);
        const DeviceCandidate* automatic = chooseDevice(candidates, {});
        for (const DeviceCandidate& candidate : candidates) {
            CopperGPUDevice device;
            device.index = candidate.index;
            device.name = candidate.properties.deviceName;
            switch (candidate.properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: device.kind = "discrete"; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: device.kind = "integrated"; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: device.kind = "virtual"; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: device.kind = "cpu"; break;
            default: device.kind = "other"; break;
            }
            device.api = "Vulkan " + std::to_string(VK_API_VERSION_MAJOR(candidate.properties.apiVersion)) + "." +
                         std::to_string(VK_API_VERSION_MINOR(candidate.properties.apiVersion));
            device.memoryBytes = candidate.deviceLocalBytes;
            device.usable = candidate.problem.empty();
            device.problem = candidate.problem;
            device.isDefault = &candidate == automatic;
            result.push_back(std::move(device));
        }
    } catch (...) {
        vkDestroyInstance(instance, nullptr);
        throw;
    }
    vkDestroyInstance(instance, nullptr);
    return result;
}

} // namespace copper

#endif // COPPER_VULKAN
