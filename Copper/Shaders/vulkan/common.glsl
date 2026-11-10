// Shared by every Copper Vulkan compute shader (see CopperVulkanEngine.cpp). Buffers are reached
// through buffer device addresses held in one Params block, so no shader needs descriptor sets.
// Field and coefficient arrays use the layout every Copper backend shares: flat, x fastest,
// index = x + nx * (y + ny * z) (copperGridIndex()).
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Uints { uint v[]; };

struct ExcitationCell {
    uint x, y, z, axis;
    float amplitude;
    uint delaySteps;
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer ExcitationCells { ExcitationCell v[]; };

struct LumpedCell {
    uint x, y, z, axis;
    float ib0, b1, b2, vv2, vj1, vj2, vvd;
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer LumpedCells { LumpedCell v[]; };

// Must match CopperVulkanEngine.cpp's VulkanParams member for member: every reference is 8 bytes,
// then the 32-bit scalars.
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Params {
    Floats e[3];
    Floats h[3];
    Floats vv[3];
    Floats vi[3];
    Floats ii[3];
    Floats iv[3];
    Uints layerOf[3]; // per axis: each grid line's CPML layer, or 0xFFFFFFFF
    Floats bE[3];
    Floats cE[3];
    Floats bH[3];
    Floats cH[3];
    Floats psi[12];   // [side * 6 + axis * 2 + term] -- side 0 is E, 1 is H
    Uints state;      // [0] = the current timestep (numTS)
    ExcitationCells voltageCells;
    ExcitationCells currentCells;
    Floats voltageSignal;
    Floats currentSignal;
    LumpedCells lumpedCells;
    Floats lumpedState;   // six per element: vdn[0..2], jn[0..2]
    Uints gatherList;     // pairs: component (0-5 = Ex..Hz), cell index
    Floats gatherOut;     // host-visible
    Floats energyPartials; // host-visible: per workgroup, E then H
    uint nx, ny, nz;
    uint cpmlActive;
    uint layerCount[3];
    int periodSteps;      // > 0 for a periodic signal; else the signal plays once
    uint voltageCount, currentCount, voltageSignalLength, currentSignalLength;
    uint lumpedCount, gatherCount;
    uint energyCellCount;
};

const uint kNoLayer = 0xFFFFFFFFu;

uint gridIndex(Params p, uint x, uint y, uint z) {
    return x + p.nx * (y + p.ny * z);
}

// Where a CPML psi value lives for grading axis `w` (CopperCPUEngine::applyCPML's `p`).
uint psiIndex(Params p, uint w, uint layer, uint x, uint y, uint z) {
    if (w == 0u) return layer + p.layerCount[0] * (y + p.ny * z);
    if (w == 1u) return x + p.nx * (layer + p.layerCount[1] * z);
    return x + p.nx * (y + p.ny * layer);
}
