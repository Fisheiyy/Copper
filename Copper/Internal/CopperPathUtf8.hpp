// UTF-8 conversions for std::filesystem::path at narrow-string boundaries (HDF5 file names, log and
// error text). On macOS and Linux a path's native narrow form is already its UTF-8 bytes, so these
// are identity conversions there. On Windows, path::string() converts to the active ANSI code page
// -- lossy, and throwing for characters it cannot represent -- whereas HDF5 (1.10.6 and later)
// interprets `const char*` file names as UTF-8.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace copper {

inline std::string pathToUtf8(const std::filesystem::path& path) {
    const std::u8string utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

inline std::filesystem::path pathFromUtf8(std::string_view utf8) {
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

} // namespace copper
