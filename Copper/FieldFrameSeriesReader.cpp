#include "FieldFrameSeriesReader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <blosc2.h>
#include <hdf5.h>

#include "Internal/CopperFieldFrameSignposts.hpp"
#include "Internal/CopperHDF5Blosc2.hpp"
#include "Internal/CopperPathUtf8.hpp"
#include "Internal/FieldFrameSampleCodec.hpp"

namespace copper {

namespace {

constexpr std::array<const char*, 6> kComponentMinNames = {
    "/frames/Ex_min", "/frames/Ey_min", "/frames/Ez_min",
    "/frames/Hx_min", "/frames/Hy_min", "/frames/Hz_min"};
constexpr std::array<const char*, 6> kComponentMaxNames = {
    "/frames/Ex_max", "/frames/Ey_max", "/frames/Ez_max",
    "/frames/Hx_max", "/frames/Hy_max", "/frames/Hz_max"};
constexpr std::array<const char*, 6> kPreviewComponentNames = {
    "/preview/Ex_mean", "/preview/Ey_mean", "/preview/Ez_mean",
    "/preview/Hx_mean", "/preview/Hy_mean", "/preview/Hz_mean"};

// Same RAII shape as FieldFrameSeriesWriter.cpp's own HId -- deliberately duplicated rather than
// shared through a private header, since these two files are the only two consumers and a shared
// tiny RAII helper isn't worth a third file just to avoid ~25 lines appearing twice.
class HId {
public:
    using Closer = herr_t (*)(hid_t);

    HId() = default;
    HId(hid_t id, Closer closer) : _id(id), _closer(closer) {}
    HId(const HId&) = delete;
    HId& operator=(const HId&) = delete;
    HId(HId&& other) noexcept : _id(other._id), _closer(other._closer) { other._id = -1; }
    HId& operator=(HId&& other) noexcept {
        if (this != &other) {
            reset();
            _id = other._id;
            _closer = other._closer;
            other._id = -1;
        }
        return *this;
    }
    ~HId() { reset(); }

    hid_t get() const { return _id; }
    bool valid() const { return _id >= 0; }
    void reset() {
        if (_id >= 0 && _closer != nullptr) {
            std::lock_guard lock(hdf5ApiMutex());
            _closer(_id);
        }
        _id = -1;
    }

private:
    hid_t _id = -1;
    Closer _closer = nullptr;
};

struct DisableHDF5ErrorPrinting {
    DisableHDF5ErrorPrinting() { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
};
const DisableHDF5ErrorPrinting kDisableHDF5ErrorPrinting;

std::expected<std::int32_t, std::string> readInt32Attribute(hid_t loc, const char* name) {
    HId attr(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose);
    if (!attr.valid()) {
        return std::unexpected(std::string("H5Aopen failed for attribute ") + name);
    }
    std::int32_t value = 0;
    if (H5Aread(attr.get(), H5T_NATIVE_INT32, &value) < 0) {
        return std::unexpected(std::string("H5Aread failed for attribute ") + name);
    }
    return value;
}

std::expected<double, std::string> readDoubleAttribute(hid_t loc, const char* name) {
    HId attr(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose);
    if (!attr.valid()) {
        return std::unexpected(std::string("H5Aopen failed for attribute ") + name);
    }
    double value = 0.0;
    if (H5Aread(attr.get(), H5T_NATIVE_DOUBLE, &value) < 0) {
        return std::unexpected(std::string("H5Aread failed for attribute ") + name);
    }
    return value;
}

std::expected<std::string, std::string> readStringAttribute(hid_t loc, const char* name) {
    HId attr(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose);
    if (!attr.valid()) {
        return std::unexpected(std::string("H5Aopen failed for attribute ") + name);
    }
    HId type(H5Aget_type(attr.get()), H5Tclose);
    if (!type.valid()) {
        return std::unexpected(std::string("H5Aget_type failed for attribute ") + name);
    }
    const std::size_t size = H5Tget_size(type.get());
    if (size == 0) {
        return std::unexpected(std::string("H5Tget_size failed for attribute ") + name);
    }
    std::string value(size, '\0');
    if (H5Aread(attr.get(), type.get(), value.data()) < 0) {
        return std::unexpected(std::string("H5Aread failed for attribute ") + name);
    }
    return value;
}

std::expected<std::vector<double>, std::string> readDoubleArrayDataset(hid_t file, const char* name,
                                                                          std::uint32_t expectedCount) {
    HId dataset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
    if (!dataset.valid()) {
        return std::unexpected(std::string("H5Dopen2 failed for ") + name);
    }
    std::vector<double> values(expectedCount);
    if (expectedCount > 0 &&
        H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
        return std::unexpected(std::string("H5Dread failed for ") + name);
    }
    return values;
}

std::expected<std::vector<std::uint8_t>, std::string> readUInt8ArrayDataset(
    hid_t file, const char* name, std::size_t expectedCount) {
    HId dataset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
    if (!dataset.valid()) return std::unexpected(std::string("H5Dopen2 failed for ") + name);
    HId space(H5Dget_space(dataset.get()), H5Sclose);
    hsize_t dims[1] = {0};
    if (!space.valid() || H5Sget_simple_extent_ndims(space.get()) != 1 ||
        H5Sget_simple_extent_dims(space.get(), dims, nullptr) < 0 || dims[0] != expectedCount) {
        return std::unexpected(std::string("Dataset extent mismatch for ") + name);
    }
    std::vector<std::uint8_t> values(expectedCount);
    if (H5Dread(dataset.get(), H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0)
        return std::unexpected(std::string("H5Dread failed for ") + name);
    return values;
}

/// Reads the published prefix of a small (one value per frame) 1D dataset eagerly. During SWMR the
/// physical dataset may already contain a later, not-yet-published block, so an extent larger than
/// frameCount is valid; only an extent smaller than the authoritative published count is corrupt.
template <typename T>
std::expected<std::vector<T>, std::string> readSmallArrayDataset(hid_t file, const char* name, hid_t nativeType,
                                                                     std::uint32_t frameCount) {
    HId dataset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
    if (!dataset.valid()) {
        return std::unexpected(std::string("H5Dopen2 failed for ") + name);
    }
    if (H5Drefresh(dataset.get()) < 0) {
        return std::unexpected(std::string("H5Drefresh failed for ") + name);
    }
    HId space(H5Dget_space(dataset.get()), H5Sclose);
    hsize_t dims[1] = {0};
    if (!space.valid() || H5Sget_simple_extent_ndims(space.get()) != 1 ||
        H5Sget_simple_extent_dims(space.get(), dims, nullptr) < 0 || dims[0] < frameCount) {
        return std::unexpected(std::string("Per-frame metadata extent mismatch for ") + name);
    }
    std::vector<T> values(frameCount);
    if (frameCount > 0) {
        const hsize_t start[1] = {0};
        const hsize_t count[1] = {frameCount};
        HId memorySpace(H5Screate_simple(1, count, nullptr), H5Sclose);
        if (!memorySpace.valid() ||
            H5Sselect_hyperslab(space.get(), H5S_SELECT_SET, start, nullptr, count, nullptr) < 0 ||
            H5Dread(dataset.get(), nativeType, memorySpace.get(), space.get(), H5P_DEFAULT,
                    values.data()) < 0) {
            return std::unexpected(std::string("H5Dread failed for ") + name);
        }
    }
    return values;
}

} // namespace

struct FieldFrameSeriesReader::Impl {
    HId file;
    // Full-resolution datasets, opened once and kept for the reader's lifetime: Ex,Ey,Ez,Hx,Hy,Hz,
    // or the single energy channel (see header.storage). The component previews and ranges below
    // exist only for a components series.
    std::vector<HId> detail;
    std::vector<std::string> detailNames;
    HId previewEnergy;
    std::array<HId, 6> previewComponent;
    HId refinementOrder;
    HId publishedFrameCountDataset;
    FieldFrameSeriesWriter::Header header;
    FieldFrameSeriesWriter::Header previewHeader;
    std::uint32_t previewFactorX = 1, previewFactorY = 1, previewFactorZ = 1;
    std::uint32_t frameCount = 0;
    std::vector<std::uint32_t> timesteps;
    std::vector<double> timeSeconds;
    std::vector<float> minEnergy;
    std::vector<float> maxEnergy;
    std::array<std::vector<float>, 6> componentMin;
    std::array<std::vector<float>, 6> componentMax;
    std::array<std::optional<FieldComponentRange>, 6> seriesComponentRange;
    mutable std::uint32_t cachedFrameIndex = std::numeric_limits<std::uint32_t>::max();
    mutable std::array<std::vector<float>, 6> cachedComponents;
    mutable std::uint32_t cachedPreviewFrameIndex = std::numeric_limits<std::uint32_t>::max();
    mutable std::vector<float> cachedPreviewEnergy;
    mutable std::array<std::vector<float>, 6> cachedPreviewComponents;
    // Second, independent slot warmed by prefetchFrame() -- kept separate so a background read of
    // the next frame never evicts the frame still being displayed.
    mutable std::uint32_t prefetchedFrameIndex = std::numeric_limits<std::uint32_t>::max();
    mutable std::array<std::vector<float>, 6> prefetchedComponents;
    mutable std::uint32_t prefetchInFlightFrameIndex = std::numeric_limits<std::uint32_t>::max();
    mutable std::uint64_t cacheGeneration = 0;
    // Guards published metadata and both cache slots. Background prefetch performs its expensive
    // decode into local storage without holding this lock, so a cached foreground read remains
    // available throughout that work.
    mutable std::mutex mutex;

    bool storesComponents() const { return header.storage.channels == FieldChannels::Components; }
    bool isSignedDetail() const { return storesComponents(); }

    /// Each detail channel's normalizing scale in `frameIndex` (see field_codec::encode()). Caller
    /// holds `mutex`; the frame must be published.
    std::vector<float> detailScalesLocked(std::uint32_t frameIndex) const {
        std::vector<float> scales;
        if (storesComponents()) {
            for (std::size_t c = 0; c < 6; ++c) {
                scales.push_back(field_codec::componentScale(componentMin[c].at(frameIndex),
                                                             componentMax[c].at(frameIndex)));
            }
        } else {
            const float peak = maxEnergy.at(frameIndex);
            scales.push_back(std::isfinite(peak) ? peak : 0.0F);
        }
        return scales;
    }

    std::expected<void, std::string> loadPublishedMetadata(std::uint32_t newFrameCount) {
        if (newFrameCount < frameCount) {
            return std::unexpected("Field-frame published count moved backwards");
        }
        for (std::size_t index = 0; index < detail.size(); ++index) {
            if (H5Drefresh(detail[index].get()) < 0) {
                return std::unexpected("H5Drefresh failed for " + detailNames[index]);
            }
            HId space(H5Dget_space(detail[index].get()), H5Sclose);
            hsize_t dims[4] = {0, 0, 0, 0};
            if (!space.valid() || H5Sget_simple_extent_dims(space.get(), dims, nullptr) < 0 ||
                dims[0] < newFrameCount) {
                return std::unexpected("Published extent is unavailable for " + detailNames[index]);
            }
        }
        if (H5Drefresh(previewEnergy.get()) < 0) {
            return std::unexpected("H5Drefresh failed for /preview/energy_max");
        }
        for (std::size_t index = 0; storesComponents() && index < previewComponent.size(); ++index) {
            if (H5Drefresh(previewComponent[index].get()) < 0) {
                return std::unexpected(std::string("H5Drefresh failed for ") + kPreviewComponentNames[index]);
            }
        }
        if (H5Drefresh(refinementOrder.get()) < 0) {
            return std::unexpected("H5Drefresh failed for /preview/refinement_order");
        }

        auto newTimesteps =
            readSmallArrayDataset<std::uint32_t>(file.get(), "/frames/timestep", H5T_NATIVE_UINT32,
                                                 newFrameCount);
        auto newTimes = readSmallArrayDataset<double>(file.get(), "/frames/time_seconds",
                                                      H5T_NATIVE_DOUBLE, newFrameCount);
        auto newMinEnergy = readSmallArrayDataset<float>(file.get(), "/frames/min_energy",
                                                         H5T_NATIVE_FLOAT, newFrameCount);
        auto newMaxEnergy = readSmallArrayDataset<float>(file.get(), "/frames/max_energy",
                                                         H5T_NATIVE_FLOAT, newFrameCount);
        if (!newTimesteps) return std::unexpected(newTimesteps.error());
        if (!newTimes) return std::unexpected(newTimes.error());
        if (!newMinEnergy) return std::unexpected(newMinEnergy.error());
        if (!newMaxEnergy) return std::unexpected(newMaxEnergy.error());

        std::array<std::vector<float>, 6> newComponentMin;
        std::array<std::vector<float>, 6> newComponentMax;
        std::array<std::optional<FieldComponentRange>, 6> newSeriesRanges;
        for (std::size_t index = 0; storesComponents() && index < newComponentMin.size(); ++index) {
            auto minima = readSmallArrayDataset<float>(file.get(), kComponentMinNames[index],
                                                       H5T_NATIVE_FLOAT, newFrameCount);
            auto maxima = readSmallArrayDataset<float>(file.get(), kComponentMaxNames[index],
                                                       H5T_NATIVE_FLOAT, newFrameCount);
            if (!minima) return std::unexpected(minima.error());
            if (!maxima) return std::unexpected(maxima.error());
            newComponentMin[index] = std::move(*minima);
            newComponentMax[index] = std::move(*maxima);
            if (newFrameCount > 0) {
                newSeriesRanges[index] = FieldComponentRange{
                    *std::min_element(newComponentMin[index].begin(), newComponentMin[index].end()),
                    *std::max_element(newComponentMax[index].begin(), newComponentMax[index].end())};
            }
        }

        timesteps = std::move(*newTimesteps);
        timeSeconds = std::move(*newTimes);
        minEnergy = std::move(*newMinEnergy);
        maxEnergy = std::move(*newMaxEnergy);
        componentMin = std::move(newComponentMin);
        componentMax = std::move(newComponentMax);
        seriesComponentRange = std::move(newSeriesRanges);
        frameCount = newFrameCount;
        return {};
    }

    std::expected<std::uint32_t, std::string> refreshPublishedFrames() {
        if (H5Drefresh(publishedFrameCountDataset.get()) < 0) {
            return std::unexpected("H5Drefresh failed for /frames/published_frame_count");
        }
        std::uint32_t publishedCount = 0;
        if (H5Dread(publishedFrameCountDataset.get(), H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, &publishedCount) < 0) {
            return std::unexpected("H5Dread failed for /frames/published_frame_count");
        }
        if (publishedCount != frameCount) {
            if (auto loaded = loadPublishedMetadata(publishedCount); !loaded) {
                return std::unexpected(loaded.error());
            }
        }
        return frameCount;
    }

    /// Decodes a full-resolution cuboid of all six components from frame `frameIndex`. Only the
    /// requested hyperslab is copied into caller-owned memory, however many frames HDF5's chunks
    /// span. HDF5 access is serialized by the caller as it requires.
    std::expected<void, std::string> decodeComponentRegion(
        std::uint32_t frameIndex, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t nx,
        std::uint32_t ny, std::uint32_t nz, const std::array<std::vector<float>*, 6>& outputs) const {
        if (!storesComponents()) {
            return std::unexpected("This field series stores only energy, not E/H components");
        }
        std::vector<float> scales;
        {
            std::lock_guard lock(mutex);
            if (frameIndex >= frameCount) return std::unexpected("Field frame is outside the published range");
            scales = detailScalesLocked(frameIndex);
        }
        const FieldSamplePrecision precision = header.storage.precision;
        const std::size_t bytesPerSample = field_codec::sampleBytes(precision);
        const hid_t storedType = field_codec::nativeType(precision, true);
        const std::size_t cellCount = static_cast<std::size_t>(nx) * ny * nz;
        const hsize_t start[4] = {frameIndex, z, y, x};
        const hsize_t count[4] = {1, nz, ny, nx};
        HId memspace(H5Screate_simple(4, count, nullptr), H5Sclose);
        if (!memspace.valid()) return std::unexpected("Could not create field detail memory space");
        const std::uint32_t previewX = x / previewFactorX;
        const std::uint32_t previewY = y / previewFactorY;
        const std::uint32_t previewZ = z / previewFactorZ;
        const std::uint32_t previewNx = (x + nx - 1) / previewFactorX - previewX + 1;
        const std::uint32_t previewNy = (y + ny - 1) / previewFactorY - previewY + 1;
        const std::uint32_t previewNz = (z + nz - 1) / previewFactorZ - previewZ + 1;
        const std::size_t previewCellCount = static_cast<std::size_t>(previewNx) * previewNy * previewNz;
        const hsize_t previewStart[4] = {frameIndex, previewZ, previewY, previewX};
        const hsize_t previewCount[4] = {1, previewNz, previewNy, previewNx};
        HId previewMemspace(H5Screate_simple(4, previewCount, nullptr), H5Sclose);
        std::vector<std::uint8_t> stored(cellCount * bytesPerSample);
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            std::vector<float> baseline(previewCellCount);
            HId previewFilespace(H5Dget_space(previewComponent[i].get()), H5Sclose);
            if (!previewMemspace.valid() || !previewFilespace.valid() ||
                H5Sselect_hyperslab(previewFilespace.get(), H5S_SELECT_SET, previewStart, nullptr,
                                    previewCount, nullptr) < 0 ||
                H5Dread(previewComponent[i].get(), H5T_NATIVE_FLOAT, previewMemspace.get(),
                        previewFilespace.get(), H5P_DEFAULT, baseline.data()) < 0) {
                return std::unexpected("Could not read field detail baseline");
            }
            outputs[i]->resize(cellCount);
            HId filespace(H5Dget_space(detail[i].get()), H5Sclose);
            if (!filespace.valid() ||
                H5Sselect_hyperslab(filespace.get(), H5S_SELECT_SET, start, nullptr, count, nullptr) < 0 ||
                H5Dread(detail[i].get(), storedType, memspace.get(), filespace.get(), H5P_DEFAULT,
                        stored.data()) < 0) {
                return std::unexpected("Could not read field detail region from " + detailNames[i]);
            }
            for (std::uint32_t localZ = 0; localZ < nz; ++localZ) {
                for (std::uint32_t localY = 0; localY < ny; ++localY) {
                    for (std::uint32_t localX = 0; localX < nx; ++localX) {
                        const std::size_t localIndex = localX + static_cast<std::size_t>(nx) *
                            (localY + static_cast<std::size_t>(ny) * localZ);
                        const std::size_t previewIndex = ((x + localX) / previewFactorX - previewX) +
                            static_cast<std::size_t>(previewNx) *
                            (((y + localY) / previewFactorY - previewY) + static_cast<std::size_t>(previewNy) *
                             ((z + localZ) / previewFactorZ - previewZ));
                        (*outputs[i])[localIndex] = field_codec::decode(
                            precision, true, stored.data() + localIndex * bytesPerSample,
                            baseline[previewIndex], scales[i]);
                    }
                }
            }
        }
        return {};
    }

    std::expected<void, std::string> decodeFrame(std::uint32_t frameIndex,
                                                  std::uint32_t availableFrameCount,
                                                  std::array<std::vector<float>, 6>& outComponents) const {
        std::lock_guard hdf5Lock(hdf5ApiMutex());
        if (frameIndex >= availableFrameCount) {
            return std::unexpected("Field frame is outside the published range");
        }
        const os_signpost_id_t signpost = os_signpost_id_generate(fieldFrameSignpostLog());
        os_signpost_interval_begin(fieldFrameSignpostLog(), signpost, "Decode field frame",
                                   "frame=%u", frameIndex);
        std::array<std::vector<float>*, 6> outputs;
        for (std::size_t i = 0; i < outputs.size(); ++i) outputs[i] = &outComponents[i];
        auto decoded = decodeComponentRegion(frameIndex, 0, 0, 0, header.nx, header.ny, header.nz, outputs);
        os_signpost_interval_end(fieldFrameSignpostLog(), signpost, "Decode field frame",
                                 "frame=%u success=%d", frameIndex, decoded ? 1 : 0);
        return decoded;
    }

    std::expected<void, std::string> decodePreviewFrame(
        std::uint32_t frameIndex, std::uint32_t availableFrameCount, std::vector<float>& energy,
        std::array<std::vector<float>*, 6> outputs) const {
        std::lock_guard hdf5Lock(hdf5ApiMutex());
        if (frameIndex >= availableFrameCount) return std::unexpected("Preview frame is outside the published range");
        const std::size_t count = static_cast<std::size_t>(previewHeader.nx) * previewHeader.ny * previewHeader.nz;
        const hsize_t start[4] = {frameIndex, 0, 0, 0};
        const hsize_t dimensions[4] = {1, previewHeader.nz, previewHeader.ny, previewHeader.nx};
        HId memspace(H5Screate_simple(4, dimensions, nullptr), H5Sclose);
        auto read = [&](hid_t dataset, float* values) -> bool {
            HId filespace(H5Dget_space(dataset), H5Sclose);
            return filespace.valid() &&
                H5Sselect_hyperslab(filespace.get(), H5S_SELECT_SET, start, nullptr, dimensions, nullptr) >= 0 &&
                H5Dread(dataset, H5T_NATIVE_FLOAT, memspace.get(), filespace.get(), H5P_DEFAULT, values) >= 0;
        };
        energy.resize(count);
        if (!memspace.valid() || !read(previewEnergy.get(), energy.data())) {
            return std::unexpected("H5Dread failed for /preview/energy_max");
        }
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (!storesComponents()) {
                outputs[i]->clear();
                continue;
            }
            outputs[i]->resize(count);
            if (!read(previewComponent[i].get(), outputs[i]->data())) {
                return std::unexpected(std::string("H5Dread failed for ") + kPreviewComponentNames[i]);
            }
        }
        return {};
    }
};

FieldFrameSeriesReader::FieldFrameSeriesReader(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}
FieldFrameSeriesReader::FieldFrameSeriesReader(FieldFrameSeriesReader&&) noexcept = default;
FieldFrameSeriesReader& FieldFrameSeriesReader::operator=(FieldFrameSeriesReader&&) noexcept = default;
FieldFrameSeriesReader::~FieldFrameSeriesReader() = default;

std::expected<FieldFrameSeriesReader, std::string> FieldFrameSeriesReader::open(const std::filesystem::path& path) {
    std::lock_guard hdf5Lock(hdf5ApiMutex());
    if (auto registered = registerHDF5Blosc2Filter(); !registered) {
        return std::unexpected(registered.error());
    }
    auto impl = std::make_unique<Impl>();
    HId fileAccess(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    if (!fileAccess.valid() ||
        H5Pset_libver_bounds(fileAccess.get(), H5F_LIBVER_LATEST, H5F_LIBVER_LATEST) < 0) {
        return std::unexpected("Could not configure the HDF5 SWMR reader");
    }
    impl->file = HId(H5Fopen(pathToUtf8(path).c_str(), H5F_ACC_RDONLY | H5F_ACC_SWMR_READ,
                             fileAccess.get()), H5Fclose);
    if (!impl->file.valid()) {
        return std::unexpected("Could not open " + pathToUtf8(path));
    }
    const hid_t file = impl->file.get();

    auto formatVersion = readInt32Attribute(file, "format_version");
    if (!formatVersion) return std::unexpected(formatVersion.error());
    if (*formatVersion != 4 && *formatVersion != 5) {
        return std::unexpected("Unsupported field frame-series format_version " + std::to_string(*formatVersion) +
                               "; rerun the simulation to create format version 5");
    }
    // Version 4 had no storage options: lossless float32 components, the default.
    if (*formatVersion >= 5) {
        auto precision = readStringAttribute(file, "sample_precision");
        auto channels = readStringAttribute(file, "channels");
        if (!precision) return std::unexpected(precision.error());
        if (!channels) return std::unexpected(channels.error());
        if (!field_codec::precisionFromName(*precision, impl->header.storage.precision)) {
            return std::unexpected("Unknown field sample precision \"" + *precision + "\"");
        }
        if (*channels == field_codec::channelsName(FieldChannels::Energy)) {
            impl->header.storage.channels = FieldChannels::Energy;
        } else if (*channels != field_codec::channelsName(FieldChannels::Components)) {
            return std::unexpected("Unknown field channels \"" + *channels + "\"");
        }
    }
    auto simulationName = readStringAttribute(file, "simulation_name");
    if (!simulationName) return std::unexpected(simulationName.error());
    impl->header.simulationName = *simulationName;
    auto excitedPort = readInt32Attribute(file, "excited_port");
    if (!excitedPort) return std::unexpected(excitedPort.error());
    impl->header.excitedPort = *excitedPort;
    auto nx = readInt32Attribute(file, "nx");
    auto ny = readInt32Attribute(file, "ny");
    auto nz = readInt32Attribute(file, "nz");
    if (!nx) return std::unexpected(nx.error());
    if (!ny) return std::unexpected(ny.error());
    if (!nz) return std::unexpected(nz.error());
    impl->header.nx = static_cast<std::uint32_t>(*nx);
    impl->header.ny = static_cast<std::uint32_t>(*ny);
    impl->header.nz = static_cast<std::uint32_t>(*nz);
    auto timestepSeconds = readDoubleAttribute(file, "timestep_seconds");
    if (!timestepSeconds) return std::unexpected(timestepSeconds.error());
    impl->header.timestepSeconds = *timestepSeconds;
    auto boardZMin = readDoubleAttribute(file, "board_z_min");
    auto boardZMax = readDoubleAttribute(file, "board_z_max");
    if (!boardZMin) return std::unexpected(boardZMin.error());
    if (!boardZMax) return std::unexpected(boardZMax.error());
    impl->header.boardZMin = *boardZMin;
    impl->header.boardZMax = *boardZMax;

    auto lineX = readDoubleArrayDataset(file, "/grid/line_x", impl->header.nx);
    auto lineY = readDoubleArrayDataset(file, "/grid/line_y", impl->header.ny);
    auto lineZ = readDoubleArrayDataset(file, "/grid/line_z", impl->header.nz);
    if (!lineX) return std::unexpected(lineX.error());
    if (!lineY) return std::unexpected(lineY.error());
    if (!lineZ) return std::unexpected(lineZ.error());
    impl->header.lineX = std::move(*lineX);
    impl->header.lineY = std::move(*lineY);
    impl->header.lineZ = std::move(*lineZ);
    auto domainXYClass = readUInt8ArrayDataset(
        file, "/grid/domain_xy_class", static_cast<std::size_t>(impl->header.nx) * impl->header.ny);
    if (!domainXYClass) return std::unexpected(domainXYClass.error());
    impl->header.domainXYClass = std::move(*domainXYClass);

    auto previewNx = readInt32Attribute(file, "preview_nx");
    auto previewNy = readInt32Attribute(file, "preview_ny");
    auto previewNz = readInt32Attribute(file, "preview_nz");
    auto factorX = readInt32Attribute(file, "preview_factor_x");
    auto factorY = readInt32Attribute(file, "preview_factor_y");
    auto factorZ = readInt32Attribute(file, "preview_factor_z");
    if (!previewNx || !previewNy || !previewNz || !factorX || !factorY || !factorZ) {
        return std::unexpected("Field frame-series preview metadata is incomplete");
    }
    if (*previewNx <= 0 || *previewNy <= 0 || *previewNz <= 0 ||
        *factorX <= 0 || *factorY <= 0 || *factorZ <= 0) {
        return std::unexpected("Field frame-series preview dimensions are invalid");
    }
    if (static_cast<std::uint64_t>(*previewNx) * static_cast<std::uint64_t>(*previewNy) *
            static_cast<std::uint64_t>(*previewNz) > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected("Field preview has too many cells for uint32 refinement indices");
    }
    impl->previewHeader = impl->header;
    impl->previewHeader.nx = static_cast<std::uint32_t>(*previewNx);
    impl->previewHeader.ny = static_cast<std::uint32_t>(*previewNy);
    impl->previewHeader.nz = static_cast<std::uint32_t>(*previewNz);
    impl->previewFactorX = static_cast<std::uint32_t>(*factorX);
    impl->previewFactorY = static_cast<std::uint32_t>(*factorY);
    impl->previewFactorZ = static_cast<std::uint32_t>(*factorZ);
    auto reducedLine = [](const std::vector<double>& full, std::uint32_t factor) {
        std::vector<double> reduced;
        reduced.reserve((full.size() + factor - 1) / factor);
        for (std::size_t begin = 0; begin < full.size(); begin += factor) {
            const std::size_t end = std::min(begin + factor, full.size());
            reduced.push_back((full[begin] + full[end - 1]) * 0.5);
        }
        return reduced;
    };
    impl->previewHeader.lineX = reducedLine(impl->header.lineX, static_cast<std::uint32_t>(*factorX));
    impl->previewHeader.lineY = reducedLine(impl->header.lineY, static_cast<std::uint32_t>(*factorY));
    impl->previewHeader.lineZ = reducedLine(impl->header.lineZ, static_cast<std::uint32_t>(*factorZ));

    impl->publishedFrameCountDataset =
        HId(H5Dopen2(file, "/frames/published_frame_count", H5P_DEFAULT), H5Dclose);
    if (!impl->publishedFrameCountDataset.valid() ||
        H5Dread(impl->publishedFrameCountDataset.get(), H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL,
                H5P_DEFAULT, &impl->frameCount) < 0) {
        return std::unexpected("Could not read /frames/published_frame_count");
    }

    // The application's own two-frame cache is the sole retention policy for large decoded data.
    // Disable HDF5's per-dataset raw chunk cache for the six component datasets: their physical
    // chunks contain several frames and allowing each open dataset to retain one would quietly
    // reintroduce many-frame residency beneath our bounded cache.
    HId componentDatasetAccess(H5Pcreate(H5P_DATASET_ACCESS), H5Pclose);
    if (!componentDatasetAccess.valid() ||
        H5Pset_chunk_cache(componentDatasetAccess.get(), 1, 1, 0.0) < 0) {
        return std::unexpected("Could not disable the field component chunk cache");
    }
    const bool storesComponents = impl->storesComponents();
    if (storesComponents) {
        for (const char* component : field_codec::kComponentNames) {
            impl->detailNames.push_back(field_codec::datasetName(impl->header.storage.precision, component));
        }
    } else {
        impl->detailNames.push_back(field_codec::datasetName(impl->header.storage.precision, "energy"));
    }
    for (std::size_t i = 0; i < impl->detailNames.size(); ++i) {
        HId dataset(H5Dopen2(file, impl->detailNames[i].c_str(), componentDatasetAccess.get()), H5Dclose);
        if (!dataset.valid()) {
            return std::unexpected("H5Dopen2 failed for " + impl->detailNames[i]);
        }
        if (i == 0) {
            // The scalar publication dataset above is authoritative. The first channel's current
            // physical extent may already include a block the writer has not published yet.
            HId space(H5Dget_space(dataset.get()), H5Sclose);
            hsize_t dims[4] = {0, 0, 0, 0};
            if (!space.valid() || H5Sget_simple_extent_dims(space.get(), dims, nullptr) < 0) {
                return std::unexpected("Could not read the extent of " + impl->detailNames[i]);
            }
            if (dims[0] < impl->frameCount) {
                return std::unexpected("Published field-frame count exceeds " + impl->detailNames[i] + "'s extent");
            }
        }
        impl->detail.push_back(std::move(dataset));
    }
    impl->previewEnergy = HId(H5Dopen2(file, "/preview/energy_max", componentDatasetAccess.get()), H5Dclose);
    if (!impl->previewEnergy.valid()) return std::unexpected("H5Dopen2 failed for /preview/energy_max");
    for (std::size_t i = 0; storesComponents && i < kPreviewComponentNames.size(); ++i) {
        impl->previewComponent[i] =
            HId(H5Dopen2(file, kPreviewComponentNames[i], componentDatasetAccess.get()), H5Dclose);
        if (!impl->previewComponent[i].valid()) {
            return std::unexpected(std::string("H5Dopen2 failed for ") + kPreviewComponentNames[i]);
        }
    }
    impl->refinementOrder =
        HId(H5Dopen2(file, "/preview/refinement_order", componentDatasetAccess.get()), H5Dclose);
    if (!impl->refinementOrder.valid()) {
        return std::unexpected("H5Dopen2 failed for /preview/refinement_order");
    }

    auto timestepValues = readSmallArrayDataset<std::uint32_t>(file, "/frames/timestep", H5T_NATIVE_UINT32, impl->frameCount);
    if (!timestepValues) return std::unexpected(timestepValues.error());
    auto timeSecondsValues = readSmallArrayDataset<double>(file, "/frames/time_seconds", H5T_NATIVE_DOUBLE, impl->frameCount);
    if (!timeSecondsValues) return std::unexpected(timeSecondsValues.error());
    auto minEnergyValues = readSmallArrayDataset<float>(file, "/frames/min_energy", H5T_NATIVE_FLOAT, impl->frameCount);
    if (!minEnergyValues) return std::unexpected(minEnergyValues.error());
    auto maxEnergyValues = readSmallArrayDataset<float>(file, "/frames/max_energy", H5T_NATIVE_FLOAT, impl->frameCount);
    if (!maxEnergyValues) return std::unexpected(maxEnergyValues.error());
    impl->timesteps = std::move(*timestepValues);
    impl->timeSeconds = std::move(*timeSecondsValues);
    impl->minEnergy = std::move(*minEnergyValues);
    impl->maxEnergy = std::move(*maxEnergyValues);

    for (std::size_t component = 0; storesComponents && component < impl->componentMin.size(); ++component) {
        auto minima = readSmallArrayDataset<float>(file, kComponentMinNames[component], H5T_NATIVE_FLOAT,
                                                   impl->frameCount);
        if (!minima) return std::unexpected(minima.error());
        auto maxima = readSmallArrayDataset<float>(file, kComponentMaxNames[component], H5T_NATIVE_FLOAT,
                                                   impl->frameCount);
        if (!maxima) return std::unexpected(maxima.error());
        impl->componentMin[component] = std::move(*minima);
        impl->componentMax[component] = std::move(*maxima);
        if (impl->frameCount > 0) {
            impl->seriesComponentRange[component] = FieldComponentRange{
                *std::min_element(impl->componentMin[component].begin(), impl->componentMin[component].end()),
                *std::max_element(impl->componentMax[component].begin(), impl->componentMax[component].end())};
        }
    }

    return FieldFrameSeriesReader(std::move(impl));
}

const FieldFrameSeriesWriter::Header& FieldFrameSeriesReader::header() const { return _impl->header; }
const FieldFrameSeriesWriter::Header& FieldFrameSeriesReader::previewHeader() const { return _impl->previewHeader; }
std::uint32_t FieldFrameSeriesReader::previewFactorX() const { return _impl->previewFactorX; }
std::uint32_t FieldFrameSeriesReader::previewFactorY() const { return _impl->previewFactorY; }
std::uint32_t FieldFrameSeriesReader::previewFactorZ() const { return _impl->previewFactorZ; }
std::expected<std::uint32_t, std::string> FieldFrameSeriesReader::refresh() const {
    // Take the process-wide HDF5 lock first, without excluding cached reads while waiting for an
    // in-progress prefetch. No other path holds the cache mutex while acquiring this lock.
    std::lock_guard hdf5Lock(hdf5ApiMutex());
    std::lock_guard lock(_impl->mutex);
    return _impl->refreshPublishedFrames();
}

std::uint32_t FieldFrameSeriesReader::frameCount() const {
    std::lock_guard lock(_impl->mutex);
    return _impl->frameCount;
}

std::expected<void, std::string> FieldFrameSeriesReader::readFrame(std::uint32_t index, std::vector<float>& ex,
                                                                       std::vector<float>& ey, std::vector<float>& ez,
                                                                       std::vector<float>& hx, std::vector<float>& hy,
                                                                       std::vector<float>& hz) const {
    Impl& impl = *_impl;
    std::array<std::vector<float>*, 6> outputs = {&ex, &ey, &ez, &hx, &hy, &hz};
    std::uint32_t availableFrameCount = 0;

    {
        std::unique_lock lock(impl.mutex);
        if (index >= impl.frameCount) {
            return std::unexpected("FieldFrameSeriesReader::readFrame: index out of range");
        }
        availableFrameCount = impl.frameCount;
        if (impl.cachedFrameIndex != index && impl.prefetchedFrameIndex == index) {
            // Already warmed by a prior prefetchFrame() call -- promote it rather than decoding.
            impl.cachedComponents = std::move(impl.prefetchedComponents);
            impl.cachedFrameIndex = index;
            impl.prefetchedFrameIndex = std::numeric_limits<std::uint32_t>::max();
        }
        if (impl.cachedFrameIndex == index) {
            for (std::size_t i = 0; i < outputs.size(); ++i) {
                *outputs[i] = impl.cachedComponents[i];
            }
            return {};
        }
    }

    // Decode outside the cache lock so the current frame remains available while HDF5 serves a miss.
    std::array<std::vector<float>, 6> decodedComponents;
    auto decoded = impl.decodeFrame(index, availableFrameCount, decodedComponents);
    if (!decoded) {
        return std::unexpected(decoded.error());
    }

    std::lock_guard lock(impl.mutex);
    impl.cachedComponents = std::move(decodedComponents);
    impl.cachedFrameIndex = index;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        *outputs[i] = impl.cachedComponents[i];
    }
    return {};
}

std::expected<void, std::string> FieldFrameSeriesReader::readPreviewFrame(
    std::uint32_t index, std::vector<float>& energy, std::vector<float>& ex, std::vector<float>& ey,
    std::vector<float>& ez, std::vector<float>& hx, std::vector<float>& hy, std::vector<float>& hz) const {
    std::uint32_t availableFrameCount = 0;
    const std::array<std::vector<float>*, 6> outputs = {&ex, &ey, &ez, &hx, &hy, &hz};
    {
        std::lock_guard lock(_impl->mutex);
        if (index >= _impl->frameCount) {
            return std::unexpected("FieldFrameSeriesReader::readPreviewFrame: index out of range");
        }
        availableFrameCount = _impl->frameCount;
        if (_impl->cachedPreviewFrameIndex == index) {
            energy = _impl->cachedPreviewEnergy;
            for (std::size_t i = 0; i < outputs.size(); ++i) *outputs[i] = _impl->cachedPreviewComponents[i];
            return {};
        }
    }
    auto decoded = _impl->decodePreviewFrame(index, availableFrameCount, energy, outputs);
    if (!decoded) return decoded;
    std::lock_guard lock(_impl->mutex);
    _impl->cachedPreviewFrameIndex = index;
    _impl->cachedPreviewEnergy = energy;
    for (std::size_t i = 0; i < outputs.size(); ++i) _impl->cachedPreviewComponents[i] = *outputs[i];
    return {};
}

std::expected<std::vector<std::uint32_t>, std::string>
FieldFrameSeriesReader::readRefinementOrder(std::uint32_t frameIndex) const {
    std::lock_guard hdf5Lock(hdf5ApiMutex());
    Impl& impl = *_impl;
    {
        std::lock_guard lock(impl.mutex);
        if (frameIndex >= impl.frameCount) {
            return std::unexpected("Field refinement-order frame is out of range");
        }
    }
    const std::size_t previewCellCount =
        static_cast<std::size_t>(impl.previewHeader.nx) * impl.previewHeader.ny * impl.previewHeader.nz;
    std::vector<std::uint32_t> order(previewCellCount);
    const hsize_t start[2] = {frameIndex, 0};
    const hsize_t count[2] = {1, previewCellCount};
    HId memspace(H5Screate_simple(2, count, nullptr), H5Sclose);
    HId filespace(H5Dget_space(impl.refinementOrder.get()), H5Sclose);
    if (!filespace.valid() || !memspace.valid() ||
        H5Sselect_hyperslab(filespace.get(), H5S_SELECT_SET, start, nullptr, count, nullptr) < 0 ||
        H5Dread(impl.refinementOrder.get(), H5T_NATIVE_UINT32, memspace.get(), filespace.get(),
                H5P_DEFAULT, order.data()) < 0) {
        return std::unexpected("Could not read /preview/refinement_order");
    }
    std::vector<bool> seen(previewCellCount, false);
    for (const std::uint32_t cell : order) {
        if (cell >= previewCellCount || seen[cell]) {
            return std::unexpected("Field refinement order is not a valid preview-cell permutation");
        }
        seen[cell] = true;
    }
    return order;
}

std::expected<void, std::string> FieldFrameSeriesReader::readPreviewCellDetail(
    std::uint32_t frameIndex, std::uint32_t previewCellIndex,
    std::vector<float>& ex, std::vector<float>& ey, std::vector<float>& ez,
    std::vector<float>& hx, std::vector<float>& hy, std::vector<float>& hz) const {
    const Impl& impl = *_impl;
    const std::uint64_t previewCellCount =
        static_cast<std::uint64_t>(impl.previewHeader.nx) * impl.previewHeader.ny * impl.previewHeader.nz;
    if (previewCellIndex >= previewCellCount) {
        return std::unexpected("Preview-cell detail index is outside the grid");
    }
    const std::uint32_t previewX = previewCellIndex % impl.previewHeader.nx;
    const std::uint32_t previewYZ = previewCellIndex / impl.previewHeader.nx;
    const std::uint32_t previewY = previewYZ % impl.previewHeader.ny;
    const std::uint32_t previewZ = previewYZ / impl.previewHeader.ny;
    const std::uint32_t x = previewX * impl.previewFactorX;
    const std::uint32_t y = previewY * impl.previewFactorY;
    const std::uint32_t z = previewZ * impl.previewFactorZ;
    return readRegion(frameIndex, x, y, z,
                      std::min(impl.previewFactorX, impl.header.nx - x),
                      std::min(impl.previewFactorY, impl.header.ny - y),
                      std::min(impl.previewFactorZ, impl.header.nz - z),
                      ex, ey, ez, hx, hy, hz);
}

std::expected<std::vector<FieldPreviewCellDetail>, std::string>
FieldFrameSeriesReader::readPreviewCellDetails(
    std::uint32_t frameIndex, const std::vector<std::uint32_t>& previewCellIndices) const {
    if (previewCellIndices.empty()) return std::vector<FieldPreviewCellDetail>{};
    Impl& impl = *_impl;
    const std::uint64_t previewCellCount =
        static_cast<std::uint64_t>(impl.previewHeader.nx) * impl.previewHeader.ny * impl.previewHeader.nz;
    {
        std::lock_guard lock(impl.mutex);
        if (frameIndex >= impl.frameCount) return std::unexpected("Field detail frame is out of range");
    }
    for (const std::uint32_t cell : previewCellIndices) {
        if (cell >= previewCellCount) return std::unexpected("Preview-cell detail index is outside the grid");
    }

    // Ensure the small preview is resident. Besides making ordinary playback cheap, this supplies
    // the exact XOR baseline for lossless reconstruction without asking HDF5 to decompress the
    // whole preview chunk again for every tile batch.
    bool previewIsCached = false;
    {
        std::lock_guard lock(impl.mutex);
        previewIsCached = impl.cachedPreviewFrameIndex == frameIndex;
    }
    if (!previewIsCached) {
        std::vector<float> energy, ex, ey, ez, hx, hy, hz;
        auto preview = readPreviewFrame(frameIndex, energy, ex, ey, ez, hx, hy, hz);
        if (!preview) return std::unexpected(preview.error());
    }

    const bool storesComponents = impl.storesComponents();
    const std::size_t channelCount = impl.detail.size();
    const FieldSamplePrecision precision = impl.header.storage.precision;
    const std::size_t bytesPerSample = field_codec::sampleBytes(precision);
    struct RawTile {
        std::vector<std::vector<std::uint8_t>> payloads;
        std::vector<std::uint32_t> filterMasks;
    };
    std::vector<RawTile> raw(previewCellIndices.size());
    std::vector<FieldPreviewCellDetail> details(previewCellIndices.size());
    std::vector<std::vector<float>> baselines(previewCellIndices.size());
    std::vector<float> scales;

    {
        std::lock_guard lock(impl.mutex);
        if (impl.cachedPreviewFrameIndex != frameIndex) {
            return std::unexpected("Preview cache changed while preparing field-detail batch");
        }
        scales = impl.detailScalesLocked(frameIndex);
        for (std::size_t tileIndex = 0; tileIndex < previewCellIndices.size(); ++tileIndex) {
            const std::uint32_t cell = previewCellIndices[tileIndex];
            for (std::size_t channel = 0; channel < channelCount; ++channel) {
                baselines[tileIndex].push_back(storesComponents ? impl.cachedPreviewComponents[channel][cell]
                                                                : impl.cachedPreviewEnergy[cell]);
            }
        }
    }

    // HDF5 object access remains serialized for SWMR safety, but only long enough to copy each
    // compressed chunk. The expensive Blosc2 decode happens below, outside this lock.
    {
        std::lock_guard hdf5Lock(hdf5ApiMutex());
        for (std::size_t tileIndex = 0; tileIndex < previewCellIndices.size(); ++tileIndex) {
            const std::uint32_t cell = previewCellIndices[tileIndex];
            const std::uint32_t px = cell % impl.previewHeader.nx;
            const std::uint32_t yz = cell / impl.previewHeader.nx;
            const std::uint32_t py = yz % impl.previewHeader.ny;
            const std::uint32_t pz = yz / impl.previewHeader.ny;
            const std::uint32_t x = px * impl.previewFactorX;
            const std::uint32_t y = py * impl.previewFactorY;
            const std::uint32_t z = pz * impl.previewFactorZ;
            details[tileIndex].previewCellIndex = cell;
            details[tileIndex].nx = std::min(impl.previewFactorX, impl.header.nx - x);
            details[tileIndex].ny = std::min(impl.previewFactorY, impl.header.ny - y);
            details[tileIndex].nz = std::min(impl.previewFactorZ, impl.header.nz - z);
            const hsize_t offset[4] = {frameIndex, z, y, x};
            raw[tileIndex].payloads.resize(channelCount);
            raw[tileIndex].filterMasks.assign(channelCount, 0);
            for (std::size_t channel = 0; channel < channelCount; ++channel) {
                const hid_t dataset = impl.detail[channel].get();
                hsize_t storedBytes = 0;
                if (H5Dget_chunk_storage_size(dataset, offset, &storedBytes) < 0 || storedBytes == 0) {
                    return std::unexpected("Could not size raw field-detail chunk from " + impl.detailNames[channel]);
                }
                auto& payload = raw[tileIndex].payloads[channel];
                payload.resize(static_cast<std::size_t>(storedBytes));
#if H5Dread_chunk_vers >= 2
                // HDF5 2.x takes the caller's buffer size and returns the number of bytes read.
                std::size_t readBytes = payload.size();
                if (H5Dread_chunk(dataset, H5P_DEFAULT, offset, &raw[tileIndex].filterMasks[channel],
                                  payload.data(), &readBytes) < 0) {
                    return std::unexpected("Could not read raw field-detail chunk from " + impl.detailNames[channel]);
                }
                payload.resize(readBytes);
#else
                // HDF5 1.x has no buffer-size parameter.  The allocation above is sized using
                // H5Dget_chunk_storage_size(), which is the documented compatibility path.
                if (H5Dread_chunk(dataset, H5P_DEFAULT, offset, &raw[tileIndex].filterMasks[channel],
                                  payload.data()) < 0) {
                    return std::unexpected("Could not read raw field-detail chunk from " + impl.detailNames[channel]);
                }
#endif
            }
        }
    }

    std::atomic<std::size_t> nextTile{0};
    std::atomic<bool> failed{false};
    std::mutex errorMutex;
    std::string error;
    const auto fail = [&](const char* message) {
        failed.store(true, std::memory_order_relaxed);
        std::lock_guard lock(errorMutex);
        error = message;
    };
    const unsigned int workerCount = std::min<unsigned int>(
        static_cast<unsigned int>(previewCellIndices.size()),
        std::max(1U, std::thread::hardware_concurrency()));
    auto worker = [&] {
        thread_local blosc2_context* context = [] {
            blosc2_dparams params = BLOSC2_DPARAMS_DEFAULTS;
            // Parallelism is across tiles here. Giving every tile another hardware-sized Blosc2
            // pool would massively oversubscribe the machine.
            params.nthreads = 1;
            return blosc2_create_dctx(params);
        }();
        std::vector<std::uint8_t> decoded;
        while (!failed.load(std::memory_order_relaxed)) {
            const std::size_t tileIndex = nextTile.fetch_add(1, std::memory_order_relaxed);
            if (tileIndex >= details.size()) break;
            auto& detail = details[tileIndex];
            const std::size_t actualCount = static_cast<std::size_t>(detail.nx) * detail.ny * detail.nz;
            for (std::size_t channel = 0; channel < channelCount && !failed.load(std::memory_order_relaxed);
                 ++channel) {
                const auto& payload = raw[tileIndex].payloads[channel];
                if ((raw[tileIndex].filterMasks[channel] & 1U) != 0) {
                    if (payload.size() % bytesPerSample != 0) {
                        fail("Unfiltered field-detail chunk has an invalid byte count");
                        break;
                    }
                    decoded = payload;
                } else {
                    std::int32_t outputBytes = 0, encodedBytes = 0;
                    if (context == nullptr ||
                        blosc2_cbuffer_sizes(payload.data(), &outputBytes, &encodedBytes, nullptr) < 0 ||
                        outputBytes <= 0 || static_cast<std::size_t>(outputBytes) % bytesPerSample != 0) {
                        fail("Could not inspect compressed field-detail chunk");
                        break;
                    }
                    decoded.resize(static_cast<std::size_t>(outputBytes));
                    const int result = blosc2_decompress_ctx(context, payload.data(), encodedBytes,
                                                             decoded.data(), outputBytes);
                    if (result != outputBytes) {
                        fail("Could not decompress field-detail chunk");
                        break;
                    }
                }
                const std::size_t decodedCount = decoded.size() / bytesPerSample;
                const bool compactEdge = decodedCount == actualCount;
                const std::uint32_t strideX = compactEdge ? detail.nx : impl.previewFactorX;
                const std::uint32_t strideY = compactEdge ? detail.ny : impl.previewFactorY;
                auto& output = storesComponents ? detail.components[channel] : detail.energy;
                output.resize(actualCount);
                for (std::uint32_t z = 0; z < detail.nz; ++z) {
                    for (std::uint32_t y = 0; y < detail.ny; ++y) {
                        for (std::uint32_t x = 0; x < detail.nx; ++x) {
                            const std::size_t destination = x + static_cast<std::size_t>(detail.nx) *
                                (y + static_cast<std::size_t>(detail.ny) * z);
                            const std::size_t source = x + static_cast<std::size_t>(strideX) *
                                (y + static_cast<std::size_t>(strideY) * z);
                            if (source >= decodedCount) {
                                fail("Decoded field-detail chunk has unexpected dimensions");
                                break;
                            }
                            output[destination] = field_codec::decode(
                                precision, storesComponents, decoded.data() + source * bytesPerSample,
                                baselines[tileIndex][channel], scales[channel]);
                        }
                    }
                }
            }
        }
    };
    std::vector<std::jthread> workers;
    workers.reserve(workerCount);
    for (unsigned int i = 0; i < workerCount; ++i) workers.emplace_back(worker);
    workers.clear(); // jthread joins here, before the failure check/result move.
    if (failed.load(std::memory_order_relaxed)) return std::unexpected(error);
    return details;
}

std::expected<void, std::string> FieldFrameSeriesReader::readRegion(
    std::uint32_t frameIndex, std::uint32_t x, std::uint32_t y, std::uint32_t z,
    std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, std::vector<float>& ex,
    std::vector<float>& ey, std::vector<float>& ez, std::vector<float>& hx,
    std::vector<float>& hy, std::vector<float>& hz) const {
    Impl& impl = *_impl;
    {
        std::lock_guard lock(impl.mutex);
        if (frameIndex >= impl.frameCount) return std::unexpected("Field detail frame is out of range");
    }
    if (nx == 0 || ny == 0 || nz == 0 || x > impl.header.nx || y > impl.header.ny || z > impl.header.nz ||
        nx > impl.header.nx - x || ny > impl.header.ny - y || nz > impl.header.nz - z) {
        return std::unexpected("Field detail region is empty or outside the grid");
    }
    std::array<std::vector<float>*, 6> outputs = {&ex, &ey, &ez, &hx, &hy, &hz};
    return impl.decodeComponentRegion(frameIndex, x, y, z, nx, ny, nz, outputs);
}

void FieldFrameSeriesReader::prefetchFrame(std::uint32_t index) const {
    Impl& impl = *_impl;
    std::uint32_t availableFrameCount = 0;
    std::uint64_t cacheGeneration = 0;
    {
        std::lock_guard lock(impl.mutex);
        if (index >= impl.frameCount) {
            return;
        }
        if (impl.cachedFrameIndex == index || impl.prefetchedFrameIndex == index ||
            impl.prefetchInFlightFrameIndex != std::numeric_limits<std::uint32_t>::max()) {
            return; // Already have it, or the one background decode slot is currently occupied.
        }
        impl.prefetchInFlightFrameIndex = index;
        availableFrameCount = impl.frameCount;
        cacheGeneration = impl.cacheGeneration;
    }

    // Decode locally while leaving the cache mutex available to foreground reads. decodeFrame()
    // separately serializes access to the shared HDF5 handles.
    std::array<std::vector<float>, 6> decodedComponents;
    auto decoded = impl.decodeFrame(index, availableFrameCount, decodedComponents);

    std::lock_guard lock(impl.mutex);
    if (impl.prefetchInFlightFrameIndex == index) {
        impl.prefetchInFlightFrameIndex = std::numeric_limits<std::uint32_t>::max();
    }
    if (decoded && cacheGeneration == impl.cacheGeneration && impl.cachedFrameIndex != index &&
        impl.prefetchedFrameIndex != index) {
        impl.prefetchedComponents = std::move(decodedComponents);
        impl.prefetchedFrameIndex = index;
    }
    // On failure, leave the prefetch slot untouched (still pointing at whatever it held before, or
    // still empty) -- the next readFrame() for this frame will decode it normally and
    // surface any real error there.
}

void FieldFrameSeriesReader::clearFrameCache() const {
    Impl& impl = *_impl;
    std::lock_guard lock(impl.mutex);
    ++impl.cacheGeneration;
    impl.cachedFrameIndex = std::numeric_limits<std::uint32_t>::max();
    impl.cachedPreviewFrameIndex = std::numeric_limits<std::uint32_t>::max();
    impl.prefetchedFrameIndex = std::numeric_limits<std::uint32_t>::max();
    // Swap with empty vectors rather than clear(): capacity is the expensive part and would
    // otherwise keep the full frame allocations resident after leaving/switching the viewer.
    for (std::vector<float>& component : impl.cachedComponents) {
        std::vector<float>().swap(component);
    }
    for (std::vector<float>& component : impl.prefetchedComponents) {
        std::vector<float>().swap(component);
    }
    std::vector<float>().swap(impl.cachedPreviewEnergy);
    for (std::vector<float>& component : impl.cachedPreviewComponents) {
        std::vector<float>().swap(component);
    }
}

std::uint32_t FieldFrameSeriesReader::timestep(std::uint32_t index) const {
    std::lock_guard lock(_impl->mutex);
    return _impl->timesteps.at(index);
}
double FieldFrameSeriesReader::timeSeconds(std::uint32_t index) const {
    std::lock_guard lock(_impl->mutex);
    return _impl->timeSeconds.at(index);
}
float FieldFrameSeriesReader::minEnergy(std::uint32_t index) const {
    std::lock_guard lock(_impl->mutex);
    return _impl->minEnergy.at(index);
}
float FieldFrameSeriesReader::maxEnergy(std::uint32_t index) const {
    std::lock_guard lock(_impl->mutex);
    return _impl->maxEnergy.at(index);
}

FieldComponentRange FieldFrameSeriesReader::componentRange(FieldComponent component, std::uint32_t index) const {
    std::lock_guard lock(_impl->mutex);
    const std::size_t componentIndex = static_cast<std::size_t>(component);
    if (!_impl->storesComponents()) return {};
    return {_impl->componentMin.at(componentIndex).at(index), _impl->componentMax.at(componentIndex).at(index)};
}

std::optional<FieldComponentRange> FieldFrameSeriesReader::componentRange(FieldComponent component) const {
    std::lock_guard lock(_impl->mutex);
    return _impl->seriesComponentRange.at(static_cast<std::size_t>(component));
}

} // namespace copper
