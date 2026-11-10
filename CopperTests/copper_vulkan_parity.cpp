// CPU-vs-Vulkan engine parity on synthetic grids: the Vulkan engine (CopperVulkanEngine.cpp) must
// reproduce CopperCPUEngine to float-rounding tolerance through every feature it implements --
// the E/H updates, CPML, voltage and current excitation (including two cells sharing one field
// value, the excitation kernel's serial path), engine-side lumped RLC, a MidStepCorrection that
// reads and writes cells between the half-steps, run()'s batched submissions, probe reads and the
// energy estimate.
//
// A plain executable rather than an XCTest case (CopperTests' own harness is Apple-only). Exits 0
// when every case passes, 1 on a mismatch, 2 when no Vulkan device is available.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "Internal/CopperCPML.hpp"
#include "Internal/CopperEngine.hpp"
#include "Internal/CopperExcitation.hpp"
#include "Internal/CopperLumpedRLC.hpp"
#include "Internal/CopperYeeGrid.hpp"

using copper::CopperEngine;
using Field = CopperEngine::Field;

namespace {

// A vacuum-like grid with mildly varying coefficients, so an indexing mistake can't hide behind
// uniform values. vi*iv stays well inside the 3D Courant limit.
copper::CopperYeeGrid makeGrid(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz) {
    copper::CopperYeeGrid grid;
    grid.dims = {nx, ny, nz};
    grid.timestepSeconds = 1e-12;
    const std::size_t n = grid.dims.cellCount();
    for (int axis = 0; axis < 3; ++axis) {
        grid.vv[axis].resize(n);
        grid.vi[axis].resize(n);
        grid.ii[axis].resize(n);
        grid.iv[axis].resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const float wobble = 0.02F * static_cast<float>((i * 7 + axis * 13) % 11) / 11.0F;
            grid.vv[axis][i] = 1.0F - 0.5F * wobble;
            grid.vi[axis][i] = 0.30F + wobble;
            grid.ii[axis][i] = 1.0F;
            grid.iv[axis][i] = 0.30F - wobble;
        }
    }
    return grid;
}

// A CPML of `depth` layers at both faces of every axis, b/c graded like a real one's.
copper::CopperCPML makeCPML(const copper::CopperGridDims& dims, std::uint32_t depth) {
    copper::CopperCPML cpml;
    const std::uint32_t lines[3] = {dims.nx, dims.ny, dims.nz};
    for (int axis = 0; axis < 3; ++axis) {
        copper::CopperCPML::Axis& a = cpml.axes[axis];
        a.layerOf.assign(lines[axis], copper::CopperCPML::kNoLayer);
        for (std::uint32_t k = 0; k < depth; ++k) {
            a.layerOf[k] = k;
            a.layerOf[lines[axis] - 1 - k] = depth + k;
        }
        for (std::uint32_t k = 0; k < 2 * depth; ++k) {
            const float grade = static_cast<float>((k % depth) + 1) / static_cast<float>(depth);
            a.bE.push_back(1.0F - 0.3F * grade);
            a.cE.push_back(-0.2F * grade);
            a.bH.push_back(1.0F - 0.25F * grade);
            a.cH.push_back(-0.15F * grade);
        }
    }
    return cpml;
}

copper::CopperExcitation makeExcitation(const copper::CopperGridDims& dims, bool sharedCell) {
    copper::CopperExcitation excitation;
    for (int i = 0; i < 120; ++i) {
        const double t = (i - 40) / 12.0;
        excitation.voltageSignal.push_back(static_cast<float>(std::exp(-t * t)));
        const double tc = (i - 39.5) / 12.0;
        excitation.currentSignal.push_back(static_cast<float>(0.3 * std::exp(-tc * tc)));
    }
    const std::uint32_t cx = dims.nx / 2, cy = dims.ny / 2, cz = dims.nz / 2;
    for (std::uint32_t z = cz - 1; z <= cz + 1; ++z) {
        excitation.voltageCells.push_back({cx, cy, z, 2, 1.5F, 0});
        excitation.voltageCells.push_back({cx + 2, cy, z, 0, -0.7F, 5});
    }
    excitation.currentCells.push_back({cx - 3, cy + 1, cz, 1, 0.4F, 3});
    if (sharedCell) {
        excitation.voltageCells.push_back({cx, cy, cz, 2, 0.25F, 7}); // same Ez as an entry above
    }
    return excitation;
}

std::vector<copper::CopperLumpedRLCCell> makeLumped(const copper::CopperGridDims& dims) {
    std::vector<copper::CopperLumpedRLCCell> cells;
    const std::uint32_t x = dims.nx / 2 + 4, y = dims.ny / 2 - 2;
    for (std::uint32_t z = dims.nz / 2 - 1; z <= dims.nz / 2; ++z) {
        copper::CopperLumpedRLCCell cell;
        cell.x = x;
        cell.y = y;
        cell.z = z;
        cell.axis = 2;
        cell.vvd = 0.8F;
        cell.vv2 = 0.1F;
        cell.vj1 = -0.05F;
        cell.vj2 = 0.02F;
        cell.ib0 = 0.3F;
        cell.b1 = -0.4F;
        cell.b2 = 0.1F;
        cells.push_back(cell);
    }
    return cells;
}

struct Probe {
    Field field;
    std::uint32_t x, y, z;
};

struct Trace {
    std::vector<float> samples; // probe values, step-major
    std::vector<double> energies;
    std::vector<float> finalFields[6];
    bool usedGPU = false;
};

enum class Mode { Sampled, MidStep, Batched };

struct Case {
    std::string name;
    copper::CopperGridDims dims;
    std::uint32_t cpmlDepth;
    bool sharedCell;
    bool lumped;
    Mode mode;
    std::uint32_t steps;
};

Trace runCase(const Case& c, CopperEngine::Backend backend) {
    const copper::CopperYeeGrid grid = makeGrid(c.dims.nx, c.dims.ny, c.dims.nz);
    const copper::CopperCPML cpml = c.cpmlDepth != 0 ? makeCPML(grid.dims, c.cpmlDepth) : copper::CopperCPML{};
    const copper::CopperExcitation excitation = makeExcitation(grid.dims, c.sharedCell);
    CopperEngine engine(grid, excitation, cpml, backend);

    Trace trace;
    trace.usedGPU = engine.currentAllocatedMetalBytes() != 0;
    const std::uint32_t mx = c.dims.nx / 2, my = c.dims.ny / 2, mz = c.dims.nz / 2;
    const std::vector<Probe> probes = {
        {Field::Ez, mx, my, mz},     {Field::Ex, mx + 3, my - 2, mz},  {Field::Hy, mx - 1, my, mz + 1},
        {Field::Hx, 1, 1, 1},        {Field::Ey, c.dims.nx - 2, 2, c.dims.nz - 3},
        {Field::Hz, mx + 4, my - 2, mz},
    };
    auto sample = [&](std::uint32_t step) {
        for (const Probe& p : probes) trace.samples.push_back(engine.readFieldCell(p.field, p.x, p.y, p.z));
        if (step % 25 == 0) trace.energies.push_back(engine.estimateEnergy());
        return true;
    };

    if (c.lumped) engine.setLumpedRLC(makeLumped(grid.dims));

    switch (c.mode) {
    case Mode::Sampled:
        engine.runWithProbeSampling(c.steps, sample);
        break;
    case Mode::MidStep: {
        // A correction that reads an E cell the first time and then every step, and writes two,
        // between the half-steps -- the CPU lumped RLC path's access pattern.
        const std::uint32_t x = mx - 4, y = my + 3, z = mz;
        CopperEngine::MidStepCorrection correction = [&] {
            const float a = engine.readFieldCell(Field::Ez, x, y, z);
            const float b = engine.readFieldCell(Field::Ez, x, y, z + 1);
            engine.writeFieldCell(Field::Ez, x, y, z, 0.9F * a + 0.05F * b);
            engine.writeFieldCell(Field::Ez, x, y, z + 1, 0.95F * b);
            // Read back a written value before the next submission.
            (void)engine.readFieldCell(Field::Ez, x, y, z);
        };
        engine.runWithProbeSampling(c.steps, sample, correction);
        break;
    }
    case Mode::Batched:
        engine.run(c.steps); // 64-step batches plus single steps
        sample(c.steps);
        engine.run(3);
        sample(c.steps + 3);
        break;
    }
    trace.energies.push_back(engine.estimateEnergy());
    for (int f = 0; f < 6; ++f) engine.readField(static_cast<Field>(f), trace.finalFields[f]);
    return trace;
}

// Largest |a-b| relative to the largest |a| (both engines run in float, so differences come only from
// rounding order and fused multiply-adds).
double relativeError(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    double scale = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        scale = std::max(scale, std::fabs(static_cast<double>(a[i])));
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return scale > 0.0 ? worst / scale : worst;
}

} // namespace

int main() {
    const std::vector<Case> cases = {
        {"interior only", {40, 36, 28}, 0, false, false, Mode::Sampled, 150},
        {"CPML + excitation", {48, 40, 32}, 6, false, false, Mode::Sampled, 250},
        {"shared excitation cell", {40, 36, 28}, 4, true, false, Mode::Sampled, 150},
        {"lumped RLC", {44, 40, 30}, 5, false, true, Mode::Sampled, 250},
        {"mid-step correction", {44, 40, 30}, 5, false, false, Mode::MidStep, 200},
        {"batched run()", {37, 29, 23}, 4, false, true, Mode::Batched, 133},
        {"odd sizes", {33, 7, 5}, 2, false, false, Mode::Sampled, 120},
    };
    // Both engines run in float with different operation orders, and the shared rounding noise
    // grows with the step count; well below anything a physics or indexing error produces.
    constexpr double kTolerance = 2e-4;

    bool allPassed = true;
    for (const Case& c : cases) {
        const Trace cpu = runCase(c, CopperEngine::Backend::CPU);
        const Trace gpu = runCase(c, CopperEngine::Backend::GPU);
        if (!gpu.usedGPU) {
            std::fprintf(stderr, "No Vulkan GPU engine available (see the message above); nothing to compare.\n");
            return 2;
        }
        double fieldError = 0.0;
        for (int f = 0; f < 6; ++f) fieldError = std::max(fieldError, relativeError(cpu.finalFields[f], gpu.finalFields[f]));
        const double probeError = relativeError(cpu.samples, gpu.samples);
        double energyError = cpu.energies.size() == gpu.energies.size() ? 0.0 : INFINITY;
        for (std::size_t i = 0; i < cpu.energies.size() && i < gpu.energies.size(); ++i) {
            const double scale = std::max(std::fabs(cpu.energies[i]), 1e-300);
            energyError = std::max(energyError, std::fabs(cpu.energies[i] - gpu.energies[i]) / scale);
        }
        double peak = 0.0;
        for (int f = 0; f < 6; ++f) {
            for (const float v : cpu.finalFields[f]) peak = std::max(peak, std::fabs(static_cast<double>(v)));
        }
        const bool passed = fieldError <= kTolerance && probeError <= kTolerance && energyError <= 1e-3 && peak > 0.0;
        allPassed = allPassed && passed;
        std::printf("%-24s %s  fields %.2e  probes %.2e  energy %.2e  (peak |field| %.3g)\n", c.name.c_str(),
                    passed ? "PASS" : "FAIL", fieldError, probeError, energyError, peak);
    }
    std::printf(allPassed ? "All cases passed.\n" : "Some cases FAILED.\n");
    return allPassed ? 0 : 1;
}
