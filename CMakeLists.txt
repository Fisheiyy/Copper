cmake_minimum_required(VERSION 3.25)

project(Copper LANGUAGES C CXX)

option(COPPER_ENABLE_METAL "Build Copper's Metal backend when available" ${APPLE})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

if(APPLE AND EXISTS "/opt/homebrew")
    list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

# Windows toolchain settings. These match what KiCad's own Windows build uses, so code compiled
# against its headers (and CSXCAD's) sees the same environment. Nothing here applies elsewhere.
if(MSVC)
    add_compile_options(/utf-8 /bigobj /Zc:__cplusplus)
    add_compile_definitions(NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)
elseif(MINGW)
    add_compile_options(-Wa,-mbig-obj)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Windows has no RPATH: executables find their DLLs next to themselves. Collect every binary in
# one directory so the tools run straight from the build tree. (A parent project may already have
# chosen a directory; keep it.)
if(WIN32 AND NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

find_package(nlohmann_json REQUIRED)
find_package(GEOS REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# Blosc2 comes from pkg-config on macOS and Linux. Windows package managers (vcpkg in particular)
# usually provide no pkg-config, but do install Blosc2's CMake package, so use that there.
set(COPPER_BLOSC2_LIBRARY "")
if(WIN32)
    find_package(Blosc2 CONFIG QUIET)
    foreach(_copper_blosc2_target IN ITEMS Blosc2::blosc2 Blosc2::blosc2_shared Blosc2::blosc2_static)
        if(TARGET ${_copper_blosc2_target})
            set(COPPER_BLOSC2_LIBRARY ${_copper_blosc2_target})
            break()
        endif()
    endforeach()
endif()
if(NOT COPPER_BLOSC2_LIBRARY)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(BLOSC2 REQUIRED IMPORTED_TARGET blosc2)
    set(COPPER_BLOSC2_LIBRARY PkgConfig::BLOSC2)
endif()

file(GLOB COPPER_UTILS_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/CopperUtils/*.cpp")
add_library(CopperUtils STATIC ${COPPER_UTILS_SOURCES})
target_include_directories(CopperUtils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/CopperUtils")
target_link_libraries(CopperUtils PUBLIC nlohmann_json::nlohmann_json GEOS::geos_c)

# CSXCAD's upstream CMake expects installed fparser and TinyXML.  Copper vendors both, so make
# those targets available under the variables its build file consumes.
add_library(CopperFParser STATIC
    submodules/fparser/fparser.cc
    submodules/fparser/fpoptimizer.cc)
target_include_directories(CopperFParser PUBLIC submodules/fparser)
target_compile_options(CopperFParser PRIVATE -w)
add_library(CopperTinyXML STATIC
    submodules/tinyxml/tinystr.cpp
    submodules/tinyxml/tinyxml.cpp
    submodules/tinyxml/tinyxmlerror.cpp
    submodules/tinyxml/tinyxmlparser.cpp)
target_include_directories(CopperTinyXML PUBLIC submodules/tinyxml)
target_compile_definitions(CopperTinyXML PRIVATE TIXML_USE_STL)
target_compile_options(CopperTinyXML PRIVATE -w)
# These static libraries are linked into CSXCAD, which is a shared library on
# Linux.  Explicit PIC is required on architectures such as AArch64.
set_target_properties(CopperUtils CopperFParser CopperTinyXML
    PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(fparser_LIBRARIES CopperFParser CACHE FILEPATH "" FORCE)
set(fparser_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodules/fparser" CACHE PATH "" FORCE)
set(TinyXML_LIBRARY CopperTinyXML CACHE FILEPATH "" FORCE)
set(TinyXML_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodules/tinyxml" CACHE PATH "" FORCE)
set(CSXCAD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# On Windows a DLL only exports what is explicitly marked, and the CopperUtils code linked into
# CSXCAD is not. Static linking also gives each executable exactly one copy of CopperUtils' logger
# state, and keeps memory CSXCAD allocates on the caller's heap.
if(WIN32)
    set(CSXCAD_LIBRARY_TYPE STATIC)
endif()
add_subdirectory(submodules/CSXCAD)
set_target_properties(CSXCAD PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED YES)
target_compile_options(CSXCAD PRIVATE -w)
target_link_libraries(CSXCAD CopperUtils)

file(GLOB_RECURSE COPPER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Copper/*.cpp")
if(APPLE AND COPPER_ENABLE_METAL)
    enable_language(OBJCXX)
    list(APPEND COPPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Copper/Internal/CopperEngine.mm")
endif()

# Copper marks no symbols for export, so on Windows (where a DLL exports only marked symbols) it is
# a static library. macOS keeps its framework, which carries the Metal library, and Linux its
# shared object.
if(WIN32)
    set(COPPER_LIBRARY_TYPE STATIC)
else()
    set(COPPER_LIBRARY_TYPE SHARED)
endif()
add_library(Copper ${COPPER_LIBRARY_TYPE} ${COPPER_SOURCES})
add_library(Copper::Copper ALIAS Copper)
target_include_directories(Copper
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Copper>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/submodules/CSXCAD/src"
        ${HDF5_INCLUDE_DIRS})
target_link_libraries(Copper
    PUBLIC CSXCAD CopperUtils nlohmann_json::nlohmann_json ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} ${COPPER_BLOSC2_LIBRARY})
if(WIN32)
    # FindHDF5's variables don't carry HDF5's usage requirements. The one that matters is
    # H5_BUILT_AS_DYNAMIC_LIB, without which MSVC cannot import H5T_NATIVE_* from a DLL.
    target_compile_definitions(Copper PUBLIC ${HDF5_DEFINITIONS})
endif()

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(Copper PRIVATE ${ACCELERATE_FRAMEWORK})
endif()

if(APPLE AND COPPER_ENABLE_METAL)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    target_link_libraries(Copper PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK})
    set_target_properties(Copper PROPERTIES FRAMEWORK TRUE FRAMEWORK_VERSION A)
    # The Objective-C++ engine is written for ARC, as the Xcode project builds it.
    target_compile_options(Copper PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    set(COPPER_METAL_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/metal")
    # newDefaultLibraryWithBundle: only looks for default.metallib in the framework's Resources.
    set(COPPER_METALLIB "${COPPER_METAL_DIRECTORY}/default.metallib")
    add_custom_command(OUTPUT "${COPPER_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${COPPER_METAL_DIRECTORY}"
        COMMAND xcrun -sdk macosx metal -c "${CMAKE_CURRENT_SOURCE_DIR}/Copper/Shaders/CopperFDTD.metal"
                -o "${COPPER_METAL_DIRECTORY}/CopperFDTD.air"
        COMMAND xcrun -sdk macosx metallib "${COPPER_METAL_DIRECTORY}/CopperFDTD.air" -o "${COPPER_METALLIB}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Copper/Shaders/CopperFDTD.metal"
                "${CMAKE_CURRENT_SOURCE_DIR}/Copper/Shaders/CopperShaderTypes.h")
    target_sources(Copper PRIVATE "${COPPER_METALLIB}")
    set_source_files_properties("${COPPER_METALLIB}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
else()
    target_compile_definitions(Copper PRIVATE COPPER_CPU_ONLY=1)
endif()

set_target_properties(Copper PROPERTIES OUTPUT_NAME Copper)

include(GNUInstallDirs)
install(TARGETS Copper CopperUtils CSXCAD
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    FRAMEWORK DESTINATION "${CMAKE_INSTALL_LIBDIR}")
