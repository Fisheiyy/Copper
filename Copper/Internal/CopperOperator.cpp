#include "CopperOperator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>

#include <CSPropExcitation.h>
#include <CSPropLumpedElement.h>
#include <CSPropMaterial.h>
#include <CSPropMetal.h>
#include <CSPrimBox.h>
#include <CSPrimLinPoly.h>
#include <CSPrimPolygon.h>
#include <CSRectGrid.h>

#include "CopperConformal.hpp"
#include "CopperPhysicalConstants.hpp"

namespace copper {

// ---- polygon scanline rasterization -- see this file's own header's top comment for why this
// exists at all (CSPrimitives::IsInside() called once per grid point was the dominant cost of
// CopperOperator's construction on a real board). Reproduces CSPrimPolygon::IsInside()'s own
// winding-number-plus-on-cartesian-edge algorithm exactly, edge by edge, in the same vertex-walk
// order (last-vertex-to-first, matching IsInside()'s own `x1,y1` starting point) -- not a
// re-derivation, a transcription, so it can be checked line-by-line against CSPrimPolygon.cpp's own
// IsInside() ----

bool CopperOperator::tryBuildPolygonRasterShape(CSPrimitives* prim, PolygonRasterShape& outShape) {
    CSPrimPolygon* poly = prim->ToPolygon();
    if (poly == nullptr) {
        poly = prim->ToLinPoly(); // CSPrimLinPoly derives from CSPrimPolygon; ToLinPoly() covers it
    }
    if (poly == nullptr) {
        return false;
    }
    // The rasterizer's callers feed it the mesh's own X/Y (axis 0/1) coordinates as the row/column
    // sweep -- only valid when those are also the polygon's own in-plane (nP,nPP) axes, i.e. its
    // normal is Z (2). libkiems never creates any other orientation (see this file's own top
    // comment), but a primitive that somehow did must fall back to the caller's plain IsInside()
    // loop rather than silently rasterizing the wrong plane.
    if (poly->GetNormDir() != 2) {
        return false;
    }
    if (poly->GetCoordInputType() != CARTESIAN) {
        return false; // IsInside()'s own TransformCoordSystem step would not be a no-op
    }
    if (poly->HasTransform()) {
        return false; // IsInside()'s own TransformCoords step would not be a no-op
    }
    const std::size_t n = poly->GetQtyCoords();
    if (n < 2) {
        return false; // CSPrimitives::IsInside() itself always returns false below this vertex count
    }
    outShape.x.resize(n);
    outShape.y.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        outShape.x[i] = poly->GetCoord(static_cast<int>(2 * i));
        outShape.y[i] = poly->GetCoord(static_cast<int>(2 * i + 1));
    }
    return true;
}

void CopperOperator::rasterizePolygonRow(const PolygonRasterShape& shape, double rowCoord,
                                           const std::vector<double>& sortedColCoords, std::vector<bool>& outInside) {
    PolygonRasterScratch scratch;
    std::vector<std::uint8_t> byteInside;
    byteInside.resize(sortedColCoords.size());
    rasterizePolygonRowBytes(shape, rowCoord, std::span<const double>(sortedColCoords), std::span<std::uint8_t>(byteInside), scratch);
    outInside.assign(byteInside.size(), false);
    for (std::size_t i = 0; i < byteInside.size(); ++i) {
        outInside[i] = byteInside[i] != 0;
    }
}

void CopperOperator::rasterizePolygonRowBytes(const PolygonRasterShape& shape, double rowCoord,
                                                std::span<const double> sortedColCoords,
                                                std::span<std::uint8_t> outInside,
                                                PolygonRasterScratch& scratch) {
    const std::size_t nc = sortedColCoords.size();
    const std::size_t np = shape.x.size();
    if (np < 2 || nc == 0) {
        std::fill(outInside.begin(), outInside.end(), 0);
        return;
    }

    // Winding-number contributions, expressed as sparse difference events over column indices.
    // Reusing the event vectors avoids clearing full-width scratch arrays for every polygon row.
    scratch.windingEvents.clear();
    int wnBaseline = 0;

    // On-cartesian-edge overrides -- CSPrimitives::IsInside()'s own two special cases for a query
    // point sitting exactly on an axis-aligned edge (a horizontal edge contributes nothing to the
    // crossing count at all, since it never satisfies `startover != endover`; a vertical edge only
    // ever matters for columns exactly at its own x). These use a second sparse event list.
    scratch.forcedEvents.clear();

    double x1 = shape.x[np - 1];
    double y1 = shape.y[np - 1];
    bool startover = (y1 >= rowCoord);
    for (std::size_t i = 0; i < np; ++i) {
        const double x2 = shape.x[i];
        const double y2 = shape.y[i];

        if ((x2 == x1) && (((rowCoord < y1) && (rowCoord > y2)) || ((rowCoord > y1) && (rowCoord < y2)))) {
            const auto first = std::lower_bound(sortedColCoords.begin(), sortedColCoords.end(), x1);
            const auto last = std::upper_bound(first, sortedColCoords.end(), x1);
            const std::size_t start = static_cast<std::size_t>(first - sortedColCoords.begin());
            const std::size_t stopEx = static_cast<std::size_t>(last - sortedColCoords.begin());
            if (start < nc) {
                scratch.forcedEvents.emplace_back(start, 1);
            }
            if (stopEx < nc) {
                scratch.forcedEvents.emplace_back(stopEx, -1);
            }
        }
        if ((y2 == y1) && (y1 == rowCoord) && (x1 != x2)) {
            const double loEdge = std::min(x1, x2);
            const double hiEdge = std::max(x1, x2);
            const auto first = std::upper_bound(sortedColCoords.begin(), sortedColCoords.end(), loEdge);
            const auto last = std::lower_bound(first, sortedColCoords.end(), hiEdge);
            const std::size_t start = static_cast<std::size_t>(first - sortedColCoords.begin());
            const std::size_t stopEx = static_cast<std::size_t>(last - sortedColCoords.begin());
            if (start < nc) {
                scratch.forcedEvents.emplace_back(start, 1);
            }
            if (stopEx < nc) {
                scratch.forcedEvents.emplace_back(stopEx, -1);
            }
        }

        const bool endover = (y2 >= rowCoord);
        if (startover != endover) {
            // y1 != y2 is guaranteed here (equal y1/y2 would force startover == endover above).
            auto condAt = [&](double xv) { return (y2 - rowCoord) * (x2 - x1) <= (y2 - y1) * (x2 - xv); };
            std::size_t lo = 0, hi = nc;
            if (y2 > y1) {
                // condAt(x) is true for a prefix of ascending sortedColCoords, false after -- find
                // the first index where it goes false, via the *exact* same comparison IsInside()
                // itself evaluates, just probed by binary search instead of once per query point.
                while (lo < hi) {
                    const std::size_t mid = lo + (hi - lo) / 2;
                    if (condAt(sortedColCoords[mid])) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                wnBaseline += 1;
                if (lo < nc) {
                    scratch.windingEvents.emplace_back(lo, -1);
                }
            } else {
                // y2 < y1: condAt(x) is false for a prefix, true after -- find the first index
                // where it goes true.
                while (lo < hi) {
                    const std::size_t mid = lo + (hi - lo) / 2;
                    if (!condAt(sortedColCoords[mid])) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                wnBaseline -= 1;
                if (lo < nc) {
                    scratch.windingEvents.emplace_back(lo, 1);
                }
            }
        }

        startover = endover;
        x1 = x2;
        y1 = y2;
    }

    std::sort(scratch.windingEvents.begin(), scratch.windingEvents.end());
    std::sort(scratch.forcedEvents.begin(), scratch.forcedEvents.end());
    int wn = wnBaseline;
    int forced = 0;
    std::size_t windingEvent = 0;
    std::size_t forcedEvent = 0;
    for (std::size_t i = 0; i < nc; ++i) {
        while (windingEvent < scratch.windingEvents.size() && scratch.windingEvents[windingEvent].first == i) {
            wn += scratch.windingEvents[windingEvent++].second;
        }
        while (forcedEvent < scratch.forcedEvents.size() && scratch.forcedEvents[forcedEvent].first == i) {
            forced += scratch.forcedEvents[forcedEvent++].second;
        }
        bool inside = (wn != 0);
        if (!inside && forced != 0) {
            inside = true;
        }
        outInside[i] = inside ? 1U : 0U;
    }
}

// ---- mesh geometry -- ported from Operator::GetDiscLine/GetDiscDelta/GetRawDiscDelta/GetYeeCoords/
// GetNodeWidth/GetNodeArea/GetEdgeLength/GetEdgeArea (operator.cpp) ----

double CopperOperator::discLine(int axis, unsigned int pos, bool dualMesh) const {
    const auto& lines = _rawLines[axis];
    if (pos >= lines.size()) {
        return 0.0;
    }
    if (!dualMesh) {
        return lines[pos];
    }
    if (pos < lines.size() - 1) {
        return 0.5 * (lines[pos] + lines[pos + 1]);
    }
    // dual node for the last line (outside the field domain) -- extrapolate by half the last cell.
    return lines[pos] + 0.5 * (lines[pos] - lines[pos - 1]);
}

double CopperOperator::rawDiscDelta(int axis, int pos) const {
    const auto& lines = _rawLines[axis];
    const auto n = static_cast<int>(lines.size());
    if (pos < 0) {
        return lines[0] - lines[1];
    }
    if (pos >= n - 1) {
        return lines[static_cast<std::size_t>(n - 2)] - lines[static_cast<std::size_t>(n - 1)];
    }
    return lines[static_cast<std::size_t>(pos + 1)] - lines[static_cast<std::size_t>(pos)];
}

double CopperOperator::discDelta(int axis, int pos, bool dualMesh) const {
    if (!dualMesh) {
        if (static_cast<unsigned int>(pos) < numLines(axis) - 1) {
            return discLine(axis, static_cast<unsigned int>(pos) + 1, false) - discLine(axis, static_cast<unsigned int>(pos), false);
        }
        return discLine(axis, static_cast<unsigned int>(pos), false) - discLine(axis, static_cast<unsigned int>(pos) - 1, false);
    }
    if (pos > 0) {
        return discLine(axis, static_cast<unsigned int>(pos), true) - discLine(axis, static_cast<unsigned int>(pos) - 1, true);
    }
    return discLine(axis, 1, false) - discLine(axis, 0, false);
}

bool CopperOperator::yeeCoords(int axis, const unsigned int pos[3], double coord[3], bool dualMesh) const {
    for (int n = 0; n < 3; ++n) {
        coord[n] = discLine(n, pos[n], dualMesh);
    }
    coord[axis] = discLine(axis, pos[axis], !dualMesh);

    if (!dualMesh) {
        if (pos[axis] >= numLines(axis) - 1) {
            return false;
        }
    } else {
        const int nP = (axis + 1) % 3;
        const int nPP = (axis + 2) % 3;
        if (pos[nP] >= numLines(nP) - 1 || pos[nPP] >= numLines(nPP) - 1) {
            return false;
        }
    }
    return true;
}

double CopperOperator::nodeWidthAt(int axis, const unsigned int pos[3], bool dualMesh) const {
    return edgeLength(axis, pos, !dualMesh);
}

double CopperOperator::nodeAreaAt(int axis, const unsigned int pos[3], bool dualMesh) const {
    const int axisP = (axis + 1) % 3;
    const int axisPP = (axis + 2) % 3;
    return nodeWidthAt(axisP, pos, dualMesh) * nodeWidthAt(axisPP, pos, dualMesh);
}

double CopperOperator::edgeLength(int axis, const unsigned int pos[3], bool dualMesh) const {
    return discDelta(axis, static_cast<int>(pos[axis]), dualMesh) * _gridDeltaMetres;
}

double CopperOperator::edgeArea(int axis, const unsigned int pos[3], bool dualMesh) const {
    return nodeAreaAt(axis, pos, dualMesh); // cartesian: GetEdgeArea is a passthrough to GetNodeArea
}

double CopperOperator::nodeWidth(int axis, const int pos[3]) const {
    // Always dualMesh=true -- the only way AverageMatQuarterCell ever calls GetNodeWidth.
    if (pos[0] < 0 || pos[1] < 0 || pos[2] < 0) {
        return 0.0;
    }
    const unsigned int upos[3] = {static_cast<unsigned int>(pos[0]), static_cast<unsigned int>(pos[1]),
                                    static_cast<unsigned int>(pos[2])};
    return nodeWidthAt(axis, upos, /*dualMesh=*/true);
}

double CopperOperator::nodeArea(int axis, const int pos[3]) const {
    if (pos[0] < 0 || pos[1] < 0 || pos[2] < 0) {
        return 0.0;
    }
    const unsigned int upos[3] = {static_cast<unsigned int>(pos[0]), static_cast<unsigned int>(pos[1]),
                                    static_cast<unsigned int>(pos[2])};
    return nodeAreaAt(axis, upos, /*dualMesh=*/true);
}

void CopperOperator::quarterCellCorner(int axis, int cornerIdx, const unsigned int pos[3], double outCoord[3]) const {
    const int nP = (axis + 1) % 3;
    const int nPP = (axis + 2) % 3;
    const double base[3] = {discLine(0, pos[0]), discLine(1, pos[1]), discLine(2, pos[2])};
    const double delta = rawDiscDelta(axis, static_cast<int>(pos[axis]));
    const double deltaP = rawDiscDelta(nP, static_cast<int>(pos[nP]));
    const double deltaPP = rawDiscDelta(nPP, static_cast<int>(pos[nPP]));
    const double deltaP_M = rawDiscDelta(nP, static_cast<int>(pos[nP]) - 1);
    const double deltaPP_M = rawDiscDelta(nPP, static_cast<int>(pos[nPP]) - 1);

    const bool right = (cornerIdx == 0 || cornerIdx == 2);
    const bool up = (cornerIdx == 0 || cornerIdx == 1);

    outCoord[axis] = base[axis] + delta * 0.5;
    outCoord[nP] = right ? base[nP] + deltaP * 0.25 : base[nP] - deltaP_M * 0.25;
    outCoord[nPP] = up ? base[nPP] + deltaPP * 0.25 : base[nPP] - deltaPP_M * 0.25;
}

void CopperOperator::halfCellTap(int axis, int tapIdx, const unsigned int pos[3], double outCoord[3]) const {
    const int nP = (axis + 1) % 3;
    const int nPP = (axis + 2) % 3;
    const double base[3] = {discLine(0, pos[0]), discLine(1, pos[1]), discLine(2, pos[2])};
    const double delta = rawDiscDelta(axis, static_cast<int>(pos[axis]));
    const double delta_M = rawDiscDelta(axis, static_cast<int>(pos[axis]) - 1);
    const double deltaP = rawDiscDelta(nP, static_cast<int>(pos[nP]));
    const double deltaPP = rawDiscDelta(nPP, static_cast<int>(pos[nPP]));

    outCoord[axis] = (tapIdx == 0) ? base[axis] - delta_M * 0.25 : base[axis] + delta * 0.25;
    outCoord[nP] = base[nP] + deltaP * 0.5;
    outCoord[nPP] = base[nPP] + deltaPP * 0.5;
}

const CopperOperator::PrimitiveTypeCache& CopperOperator::primitiveTypeCache(CSProperties::PropertyType type) const {
    const auto key = static_cast<unsigned int>(type);
    auto it = _primitiveTypeCache.find(key);
    if (it == _primitiveTypeCache.end()) {
        PrimitiveTypeCache cache;
        cache.prims = _csx.GetAllPrimitives(true, type);
        cache.boundBoxes.resize(cache.prims.size());
        for (std::size_t i = 0; i < cache.prims.size(); ++i) {
            // GetBoundBox()'s return value means "this box is exact", not "this box is valid":
            // CSPrimPolygon/CSPrimLinPoly always return false while still filling a correct,
            // conservative box. Treating false as unusable silently dropped every polygon-shaped
            // material (substrates, solder mask, via fill, NPTH voids) from the grid. A box is usable
            // when it was filled in (finite) and has extent on at least one axis; the base class
            // leaves it untouched (NaN) and a degenerate polygon zeroes it.
            auto& entry = cache.boundBoxes[i];
            const bool exact = cache.prims[i]->GetBoundBox(entry.box.data());
            const bool finite = std::all_of(entry.box.begin(), entry.box.end(), [](double v) { return std::isfinite(v); });
            const bool extent = entry.box[0] != entry.box[1] || entry.box[2] != entry.box[3] || entry.box[4] != entry.box[5];
            entry.ok = exact || (finite && extent);
        }
        it = _primitiveTypeCache.emplace(key, std::move(cache)).first;
    }
    return it->second;
}

std::vector<std::size_t> CopperOperator::primitivesBoundBoxIndices(int posX, int posY, int posZ,
                                                                      CSProperties::PropertyType type) const {
    double boundBox[6];
    const int bbPos[3] = {posX, posY, posZ};
    for (int n = 0; n < 3; ++n) {
        const unsigned int nLines = numLines(n);
        if (bbPos[n] < 0) {
            boundBox[2 * n] = discLine(n, 0);
            boundBox[2 * n + 1] = discLine(n, nLines - 1);
        } else {
            const unsigned int lo = bbPos[n] > 0 ? static_cast<unsigned int>(bbPos[n]) - 1 : 0;
            const unsigned int hi = std::min(nLines - 1, static_cast<unsigned int>(bbPos[n]) + 1);
            boundBox[2 * n] = discLine(n, lo);
            boundBox[2 * n + 1] = discLine(n, hi);
        }
    }
    // Filters the once-built-per-type cache (see primitiveTypeCache()'s own doc comment) instead of
    // calling ContinuousStructure::GetPrimitivesByBoundBox(..., sorted=true, ...) directly, which
    // would re-sort every primitive of `type` from scratch on every call -- IsInsideBox() itself is
    // the only genuinely per-column work here, matching GetPrimitivesByBoundBox()'s own filtering
    // step exactly (same predicate, same order-preserving scan, so this returns byte-for-byte the
    // same list openEMS's own PaintMaterialColumn/PaintPECColumn would have gotten).
    const PrimitiveTypeCache& cache = primitiveTypeCache(type);
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < cache.prims.size(); ++i) {
        if (cache.prims[i]->IsInsideBox(boundBox) >= 0) {
            out.push_back(i);
        }
    }
    return out;
}

std::vector<CSPrimitives*> CopperOperator::primitivesBoundBox(int posX, int posY, int posZ,
                                                                 CSProperties::PropertyType type) const {
    const PrimitiveTypeCache& cache = primitiveTypeCache(type);
    std::vector<CSPrimitives*> out;
    for (std::size_t i : primitivesBoundBoxIndices(posX, posY, posZ, type)) {
        out.push_back(cache.prims[i]);
    }
    return out;
}

// ---- mesh snapping -- ported from Operator::SnapToMeshLine/SnapToMesh/SnapBox2Mesh ----

unsigned int CopperOperator::snapToMeshLine(int axis, double coord, bool& inside, bool dualMesh) const {
    inside = false;
    const unsigned int n = numLines(axis);
    if (coord < discLine(axis, 0)) {
        return 0;
    }
    if (coord > discLine(axis, n - 1)) {
        return n - 1;
    }
    inside = true;
    if (!dualMesh) {
        for (unsigned int i = 0; i < n; ++i) {
            if (coord <= discLine(axis, i, true)) {
                return i;
            }
        }
    } else {
        for (unsigned int i = 1; i < n; ++i) {
            if (coord <= discLine(axis, i, false)) {
                return i - 1;
            }
        }
    }
    return 0; // should not happen
}

bool CopperOperator::snapToMesh(const double coord[3], unsigned int uicoord[3], bool dualMesh, bool* inside) const {
    bool ok = true;
    for (int n = 0; n < 3; ++n) {
        bool meshInside = false;
        uicoord[n] = snapToMeshLine(n, coord[n], meshInside, dualMesh);
        ok = ok && meshInside;
        if (inside != nullptr) {
            inside[n] = meshInside;
        }
    }
    return ok;
}

int CopperOperator::snapBox2Mesh(const double start[3], const double stop[3], unsigned int uiStart[3],
                                   unsigned int uiStop[3], bool dualMesh, int snapMethod, bool* startIn,
                                   bool* stopIn) const {
    double lStart[3], lStop[3];
    for (int n = 0; n < 3; ++n) {
        lStart[n] = std::min(start[n], stop[n]);
        lStop[n] = std::max(start[n], stop[n]);
        const double lo = discLine(n, 0);
        const double hi = discLine(n, numLines(n) - 1);
        if ((lStart[n] < lo && lStop[n] < lo) || (lStart[n] > hi && lStop[n] > hi)) {
            return -2;
        }
    }

    snapToMesh(lStart, uiStart, dualMesh, startIn);
    snapToMesh(lStop, uiStop, dualMesh, stopIn);
    int dims = 0;

    if (snapMethod == 0) {
        for (int n = 0; n < 3; ++n) {
            if (uiStop[n] > uiStart[n]) {
                ++dims;
            }
        }
        return dims;
    }
    if (snapMethod == 1) {
        for (int n = 0; n < 3; ++n) {
            if (uiStop[n] > uiStart[n]) {
                if (discLine(n, uiStart[n], dualMesh) > lStart[n] && uiStart[n] > 0) {
                    --uiStart[n];
                }
                if (discLine(n, uiStop[n], dualMesh) < lStop[n] && uiStop[n] < numLines(n) - 1) {
                    ++uiStop[n];
                }
            }
            if (uiStop[n] > uiStart[n]) {
                ++dims;
            }
        }
        return dims;
    }
    if (snapMethod == 2) {
        for (int n = 0; n < 3; ++n) {
            if (uiStop[n] > uiStart[n]) {
                if (discLine(n, uiStart[n], dualMesh) < lStart[n] && uiStart[n] < numLines(n) - 1) {
                    ++uiStart[n];
                }
                if (discLine(n, uiStop[n], dualMesh) > lStop[n] && uiStop[n] > 0) {
                    --uiStop[n];
                }
            }
            if (uiStop[n] > uiStart[n]) {
                ++dims;
            }
        }
        return dims;
    }
    return -1;
}

// ---- material/PEC resolution -- ported from Operator::PaintMaterialColumn/Calc_EC_Range/
// AverageMatQuarterCell/PaintPECColumn/CalcPEC_Range/ApplyElectricBC ----

void CopperOperator::quarterCellAverage(int axis, const unsigned int pos[3], double effMat[4],
                                         const std::vector<const ResolvedMaterial*> rowMatCache[3][6],
                                         const std::array<double, 12>& background,
                                         const MaterialGeometry& geometry) const {
    const std::size_t n = static_cast<std::size_t>(axis);
    const std::size_t nP = (n + 1) % 3;
    const std::size_t nPP = (n + 2) % 3;
    double area = 0.0;
    const std::size_t rowIndex = static_cast<std::size_t>(pos[0]) * numLines(2) + pos[2];
    const ResolvedMaterial* material0 = rowMatCache[n][0].data()[rowIndex];
    const ResolvedMaterial* material1 = rowMatCache[n][1].data()[rowIndex];
    const ResolvedMaterial* material2 = rowMatCache[n][2].data()[rowIndex];
    const ResolvedMaterial* material3 = rowMatCache[n][3].data()[rowIndex];
    const ResolvedMaterial* material4 = rowMatCache[n][4].data()[rowIndex];
    const ResolvedMaterial* material5 = rowMatCache[n][5].data()[rowIndex];
    const auto& values0 = material0 == nullptr ? background : material0->values;
    const auto& values1 = material1 == nullptr ? background : material1->values;
    const auto& values2 = material2 == nullptr ? background : material2->values;
    const auto& values3 = material3 == nullptr ? background : material3->values;
    const auto& values4 = material4 == nullptr ? background : material4->values;
    const auto& values5 = material5 == nullptr ? background : material5->values;
    const std::size_t materialOffset = n * 4;
    const double* primaryN = geometry.primaryWidth[n].data();
    const double* primaryNP = geometry.primaryWidth[nP].data();
    const double* primaryNPP = geometry.primaryWidth[nPP].data();
    const unsigned int posNP = pos[nP];
    const unsigned int posNPP = pos[nPP];
    const double widthNP = primaryNP[posNP];
    const double widthNPP = primaryNPP[posNPP];
    const double widthNPLow = posNP == 0 ? 0.0 : primaryNP[posNP - 1];
    const double widthNPPLow = posNPP == 0 ? 0.0 : primaryNPP[posNPP - 1];

    // epsilon, kappa averaging (4 quarter-cell corners)
    double aN;
    aN = widthNP * widthNPP;
    effMat[0] = values0[materialOffset] * aN;
    effMat[1] = values0[materialOffset + 1] * aN;
    area += aN;

    aN = widthNPLow * widthNPP;
    effMat[0] += values1[materialOffset] * aN;
    effMat[1] += values1[materialOffset + 1] * aN;
    area += aN;

    aN = widthNP * widthNPPLow;
    effMat[0] += values2[materialOffset] * aN;
    effMat[1] += values2[materialOffset + 1] * aN;
    area += aN;

    aN = widthNPLow * widthNPPLow;
    effMat[0] += values3[materialOffset] * aN;
    effMat[1] += values3[materialOffset + 1] * aN;
    area += aN;

    effMat[0] *= physical::epsilon0 / area;
    effMat[1] /= area;

    // mu, sigma averaging (2 half-cell taps)
    double length = 0.0;

    double deltaNy = pos[n] == 0 ? 0.0 : primaryN[pos[n] - 1];
    effMat[2] = deltaNy / values4[materialOffset + 2];
    double sigma = values4[materialOffset + 3];
    effMat[3] = (sigma != 0.0) ? deltaNy / sigma : 0.0;
    length = deltaNy;

    deltaNy = primaryN[pos[n]];
    effMat[2] += deltaNy / values5[materialOffset + 2];
    sigma = values5[materialOffset + 3];
    if (sigma != 0.0) {
        effMat[3] += deltaNy / sigma;
    } else {
        effMat[3] = 0.0;
    }
    length += deltaNy;

    effMat[2] = length * physical::mu0 / effMat[2];
    if (effMat[3] != 0.0) {
        effMat[3] = length / effMat[3];
    }
}

void CopperOperator::computeMaterialCoefficients() {
    const unsigned int nx = numLines(0);
    const unsigned int ny = numLines(1);
    const unsigned int nz = numLines(2);
    const std::size_t cellCount = _grid.dims.cellCount();
    for (int a = 0; a < 3; ++a) {
        _ecC[a].assign(cellCount, 0.0);
        _ecG[a].assign(cellCount, 0.0);
        _ecL[a].assign(cellCount, 0.0);
        _ecR[a].assign(cellCount, 0.0);
    }

    double maxCellWidthX = 0.0;
    for (unsigned int x = 0; x < nx; ++x) {
        maxCellWidthX = std::max(maxCellWidthX, std::abs(rawDiscDelta(0, static_cast<int>(x))));
    }
    double maxCellWidthZ = 0.0;
    for (unsigned int z = 0; z < nz; ++z) {
        maxCellWidthZ = std::max(maxCellWidthZ, std::abs(rawDiscDelta(2, static_cast<int>(z))));
    }
    const auto& xLines = _rawLines[0];
    const auto& zLines = _rawLines[2];

    // quarterCellCorner(m,cornerIdx,pos,_)/halfCellTap(m,tapIdx,pos,_)'s outCoord[k] depends only on
    // pos[k] (for whichever mesh axis k the caller asks about), via one of exactly 3 formula shapes
    // (see arrayFor() below) -- never on the *other* two position components. So instead of deriving
    // one coordinate at a time from a live (x,y,z) triple, precompute each formula shape as a full
    // array over every line index on every mesh axis, once, and let the rasterizer below sweep an
    // entire row of x values (and every z in a primitive's own range) by reading pre-tabulated
    // scalars instead of calling quarterCellCorner()/halfCellTap() at all -- see this file's own top
    // comment for why per-point IsInside() (which this replaces) was the dominant cost otherwise.
    MaterialGeometry geometry;
    for (int m = 0; m < 3; ++m) {
        const unsigned int n = numLines(m);
        geometry.center[static_cast<std::size_t>(m)].resize(n);
        geometry.quarterRight[static_cast<std::size_t>(m)].resize(n);
        geometry.quarterLeft[static_cast<std::size_t>(m)].resize(n);
        geometry.primaryWidth[static_cast<std::size_t>(m)].resize(n);
        geometry.dualWidth[static_cast<std::size_t>(m)].resize(n);
        for (unsigned int p = 0; p < n; ++p) {
            const double base = discLine(m, p);
            const double delta = rawDiscDelta(m, static_cast<int>(p));
            const double deltaM = rawDiscDelta(m, static_cast<int>(p) - 1);
            geometry.center[static_cast<std::size_t>(m)][p] = base + delta * 0.5;
            geometry.quarterRight[static_cast<std::size_t>(m)][p] = base + delta * 0.25;
            geometry.quarterLeft[static_cast<std::size_t>(m)][p] = base - deltaM * 0.25;
            geometry.primaryWidth[static_cast<std::size_t>(m)][p] =
                discDelta(m, static_cast<int>(p), false) * _gridDeltaMetres;
            geometry.dualWidth[static_cast<std::size_t>(m)][p] = discDelta(m, static_cast<int>(p), true) * _gridDeltaMetres;
        }
    }
    // `coordinateTable[yeeAxis][kind][meshAxis]` is the pre-tabulated contribution to one
    // quarter-corner/half-tap coordinate. Build the small dispatch table once rather than deciding
    // which vector to read in every primitive/axis/kind rasterization pass.
    const std::vector<double>* coordinateTable[3][6][3];
    for (int yeeAxis = 0; yeeAxis < 3; ++yeeAxis) {
        const int nP = (yeeAxis + 1) % 3;
        for (int kind = 0; kind < 6; ++kind) {
            for (int meshAxis = 0; meshAxis < 3; ++meshAxis) {
                const std::size_t meshAxisIndex = static_cast<std::size_t>(meshAxis);
                if (kind < 4) {
                    const bool right = (kind == 0 || kind == 2);
                    const bool up = (kind == 0 || kind == 1);
                    if (meshAxis == yeeAxis) {
                        coordinateTable[yeeAxis][kind][meshAxis] = &geometry.center[meshAxisIndex];
                    } else if (meshAxis == nP) {
                        coordinateTable[yeeAxis][kind][meshAxis] =
                            right ? &geometry.quarterRight[meshAxisIndex] : &geometry.quarterLeft[meshAxisIndex];
                    } else {
                        coordinateTable[yeeAxis][kind][meshAxis] =
                            up ? &geometry.quarterRight[meshAxisIndex] : &geometry.quarterLeft[meshAxisIndex];
                    }
                } else if (meshAxis == yeeAxis) {
                    coordinateTable[yeeAxis][kind][meshAxis] =
                        kind == 4 ? &geometry.quarterLeft[meshAxisIndex] : &geometry.quarterRight[meshAxisIndex];
                } else {
                    coordinateTable[yeeAxis][kind][meshAxis] = &geometry.center[meshAxisIndex];
                }
            }
        }
    }

    // Finds the [start,stopEx) line-index range (padded one line each side) that could contain
    // `lo..hi` along `lines` -- matches every other bbox-to-index-range conversion in this file.
    auto boundedRange = [](const std::vector<double>& lines, double lo, double hi,
                            unsigned int n) -> std::pair<unsigned int, unsigned int> {
        auto start = static_cast<unsigned int>(std::lower_bound(lines.begin(), lines.end(), lo) - lines.begin());
        auto stopEx = static_cast<unsigned int>(std::upper_bound(lines.begin(), lines.end(), hi) - lines.begin());
        if (start > 0) {
            --start;
        }
        if (stopEx < n) {
            ++stopEx;
        }
        return {start, stopEx};
    };

    const PrimitiveTypeCache& materialTypeCache = primitiveTypeCache(CSProperties::MATERIAL);
    std::vector<ResolvedMaterial> resolvedMaterials;
    resolvedMaterials.reserve(materialTypeCache.prims.size());
    std::unordered_map<CSPropMaterial*, std::size_t> materialIndices;
    std::vector<const ResolvedMaterial*> primitiveMaterials(materialTypeCache.prims.size(), nullptr);
    for (std::size_t i = 0; i < materialTypeCache.prims.size(); ++i) {
        auto* material = dynamic_cast<CSPropMaterial*>(materialTypeCache.prims[i]->GetProperty());
        if (material == nullptr) {
            continue;
        }
        // Multiple primitives commonly share one property, so resolve each property only once.
        const auto existing = materialIndices.find(material);
        if (existing != materialIndices.end()) {
            primitiveMaterials[i] = &resolvedMaterials[existing->second];
            continue;
        }
        ResolvedMaterial resolved;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const int materialAxis = static_cast<int>(axis);
            resolved.values[axis * 4] = material->GetEpsilon(materialAxis);
            resolved.values[axis * 4 + 1] = material->GetKappa(materialAxis);
            resolved.values[axis * 4 + 2] = material->GetMue(materialAxis);
            resolved.values[axis * 4 + 3] = material->GetSigma(materialAxis);
        }
        resolvedMaterials.push_back(resolved);
        materialIndices.emplace(material, resolvedMaterials.size() - 1);
        primitiveMaterials[i] = &resolvedMaterials.back();
    }
    CSBackgroundMaterial* backgroundMaterial = _csx.GetBackgroundMaterial();
    const std::array<double, 4> backgroundValues = {backgroundMaterial->GetEpsilon(), backgroundMaterial->GetKappa(),
                                                     backgroundMaterial->GetMue(), backgroundMaterial->GetSigma()};
    std::array<double, 12> background;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t value = 0; value < 4; ++value) {
            background[axis * 4 + value] = backgroundValues[value];
        }
    }

    // rowMatCache[axis][kind][x*nz+z] -- one row's worth of matCache below, widened to cover every
    // column in the row at once (row-major, not column-major, is what lets a CSPrimPolygon/
    // CSPrimLinPoly's coverage be rasterized once per row instead of tested point by point).
    std::vector<const ResolvedMaterial*> rowMatCache[3][6];
    for (int a = 0; a < 3; ++a) {
        for (int k = 0; k < 6; ++k) {
            rowMatCache[a][k].assign(static_cast<std::size_t>(nx) * nz, nullptr);
        }
    }

    PolygonRasterShape shape;
    std::vector<std::uint8_t> insideMask(nx);
    PolygonRasterScratch rasterScratch;

    for (unsigned int y = 0; y < ny; ++y) {
        // -- PaintMaterialColumn, widened to a whole row --
        for (int a = 0; a < 3; ++a) {
            for (int k = 0; k < 6; ++k) {
                std::fill(rowMatCache[a][k].begin(), rowMatCache[a][k].end(), nullptr);
            }
        }

        std::vector<std::size_t> vPrimIdx = primitivesBoundBoxIndices(-1, static_cast<int>(y), -1, CSProperties::MATERIAL);

        for (auto it = vPrimIdx.rbegin(); it != vPrimIdx.rend(); ++it) {
            CSPrimitives* prim = materialTypeCache.prims[*it];
            const ResolvedMaterial* material = primitiveMaterials[*it];
            if (material == nullptr) {
                continue;
            }
            const BoundBoxEntry& bbEntry = materialTypeCache.boundBoxes[*it];
            if (!bbEntry.ok) {
                continue;
            }
            const double* bb = bbEntry.box.data();

            const double zLo = std::min(bb[4], bb[5]) - maxCellWidthZ;
            const double zHi = std::max(bb[4], bb[5]) + maxCellWidthZ;
            const auto [zStart, zStopEx] = boundedRange(zLines, zLo, zHi, nz);
            const double xLo = std::min(bb[0], bb[1]) - maxCellWidthX;
            const double xHi = std::max(bb[0], bb[1]) + maxCellWidthX;
            const auto [xStart, xStopEx] = boundedRange(xLines, xLo, xHi, nx);
            if (xStart >= xStopEx || zStart >= zStopEx) {
                continue;
            }

            const bool canRasterize = tryBuildPolygonRasterShape(prim, shape);
            for (int axis = 0; axis < 3; ++axis) {
                for (int kind = 0; kind < 6; ++kind) {
                    if (canRasterize) {
                        const double rowCoord = (*coordinateTable[axis][kind][1])[y];
                        const std::vector<double>& fullCols = *coordinateTable[axis][kind][0];
                        const auto cols = std::span<const double>(fullCols).subspan(xStart, xStopEx - xStart);
                        const auto mask = std::span<std::uint8_t>(insideMask).first(cols.size());
                        rasterizePolygonRowBytes(shape, rowCoord, cols, mask, rasterScratch);

                        const std::vector<double>& zArray = *coordinateTable[axis][kind][2];
                        for (unsigned int z = zStart; z < zStopEx; ++z) {
                            const double zc = zArray[z];
                            if (zc < bb[4] || zc > bb[5]) {
                                continue; // IsInside()'s own elevation-axis bbox check, done
                                          // separately -- see rasterizePolygonRow()'s doc comment.
                            }
                            for (unsigned int xi = 0; xi < mask.size(); ++xi) {
                                if (mask[xi]) {
                                    rowMatCache[axis][kind][static_cast<std::size_t>(xStart + xi) * nz + z] = material;
                                }
                            }
                        }
                    } else {
                        unsigned int pos[3] = {0, y, 0};
                        double coord[3];
                        for (unsigned int z = zStart; z < zStopEx; ++z) {
                            pos[2] = z;
                            for (unsigned int x = xStart; x < xStopEx; ++x) {
                                pos[0] = x;
                                if (kind < 4) {
                                    quarterCellCorner(axis, kind, pos, coord);
                                } else {
                                    halfCellTap(axis, kind - 4, pos, coord);
                                }
                                if (prim->IsInside(coord)) {
                                    rowMatCache[axis][kind][static_cast<std::size_t>(x) * nz + z] = material;
                                }
                            }
                        }
                    }
                }
            }
        }

        // -- Calc_EC_Range for this row --
        for (unsigned int x = 0; x < nx; ++x) {
            unsigned int pos[3] = {x, y, 0};
            for (pos[2] = 0; pos[2] < nz; ++pos[2]) {
                const std::size_t i = index(pos[0], pos[1], pos[2]);
                for (int n = 0; n < 3; ++n) {
                    const std::size_t axis = static_cast<std::size_t>(n);
                    const std::size_t axisP = (axis + 1) % 3;
                    const std::size_t axisPP = (axis + 2) % 3;
                    double effMat[4];
                    quarterCellAverage(n, pos, effMat, rowMatCache, background, geometry);

                    double delta = geometry.primaryWidth[axis][pos[axis]];
                    double area = geometry.dualWidth[axisP][pos[axisP]] * geometry.dualWidth[axisPP][pos[axisPP]];
                    if (delta != 0.0) {
                        _ecC[n][i] = effMat[0] * area / delta;
                        _ecG[n][i] = effMat[1] * area / delta;
                    }

                    delta = geometry.dualWidth[axis][pos[axis]];
                    area = geometry.primaryWidth[axisP][pos[axisP]] * geometry.primaryWidth[axisPP][pos[axisPP]];
                    if (delta != 0.0) {
                        _ecL[n][i] = effMat[2] * area / delta;
                        _ecR[n][i] = effMat[3] * area / delta;
                    }
                }
            }
        }
    }
}

void CopperOperator::computeBoundaryPEC() {
    const unsigned int nLines[3] = {numLines(0), numLines(1), numLines(2)};
    for (int n = 0; n < 3; ++n) {
        const int nP = (n + 1) % 3;
        const int nPP = (n + 2) % 3;
        const bool pecLow = _config.boundary[static_cast<std::size_t>(2 * n)] == BoundaryType::PEC;
        const bool pecHigh = _config.boundary[static_cast<std::size_t>(2 * n + 1)] == BoundaryType::PEC;
        if (!pecLow && !pecHigh) {
            continue;
        }
        unsigned int pos[3];
        for (pos[nP] = 0; pos[nP] < nLines[nP]; ++pos[nP]) {
            for (pos[nPP] = 0; pos[nPP] < nLines[nPP]; ++pos[nPP]) {
                if (pecLow) {
                    pos[n] = 0;
                    const std::size_t i = index(pos[0], pos[1], pos[2]);
                    _grid.vv[nP][i] = 0.0F;
                    _grid.vi[nP][i] = 0.0F;
                    _grid.vv[nPP][i] = 0.0F;
                    _grid.vi[nPP][i] = 0.0F;
                }
                if (pecHigh) {
                    pos[n] = nLines[n] - 1;
                    const std::size_t i = index(pos[0], pos[1], pos[2]);
                    _grid.vv[n][i] = 0.0F; // outside the FDTD domain proper, matching ApplyElectricBC's own comment
                    _grid.vi[n][i] = 0.0F;
                    _grid.vv[nP][i] = 0.0F;
                    _grid.vi[nP][i] = 0.0F;
                    _grid.vv[nPP][i] = 0.0F;
                    _grid.vi[nPP][i] = 0.0F;
                }
            }
        }
    }
}

void CopperOperator::computePEC() {
    const unsigned int nx = numLines(0);
    const unsigned int ny = numLines(1);
    const unsigned int nz = numLines(2);
    const auto& xLines = _rawLines[0];
    const auto& zLines = _rawLines[2];

    // coord[] below only ever needs the primary- or dual-mesh line value for the *current* x, y, or
    // z -- none of which depend on which primitive is being tested, or (for x/y) even on z -- so
    // hoist every discLine() call out of the (x,y,z,axis,primitive) loop nest rather than re-deriving
    // the same handful of scalars on every single call.
    std::vector<double> xPrimary(nx), xDual(nx), yPrimary(ny), yDual(ny), zPrimary(nz), zDual(nz);
    for (unsigned int i = 0; i < nx; ++i) {
        xPrimary[i] = discLine(0, i, false);
        xDual[i] = discLine(0, i, true);
    }
    for (unsigned int i = 0; i < ny; ++i) {
        yPrimary[i] = discLine(1, i, false);
        yDual[i] = discLine(1, i, true);
    }
    for (unsigned int i = 0; i < nz; ++i) {
        zPrimary[i] = discLine(2, i, false);
        zDual[i] = discLine(2, i, true);
    }

    // Finds the [start,stopEx) line-index range (padded by one line each side, matching every other
    // bbox-to-index-range conversion in this file) that could contain `lo..hi` along `lines` --
    // `bounded=false` (a non-finite/unusable bbox axis) means "the whole domain", matching every
    // caller's own existing fallback.
    auto boundedRange = [](const std::vector<double>& lines, double lo, double hi, bool bounded,
                            unsigned int n) -> std::pair<unsigned int, unsigned int> {
        if (!bounded) {
            return {0, n};
        }
        auto start = static_cast<unsigned int>(std::lower_bound(lines.begin(), lines.end(), lo) - lines.begin());
        auto stopEx = static_cast<unsigned int>(std::upper_bound(lines.begin(), lines.end(), hi) - lines.begin());
        if (start > 0) {
            --start;
        }
        if (stopEx < n) {
            ++stopEx;
        }
        return {start, stopEx};
    };

    const CSProperties::PropertyType pecType =
        static_cast<CSProperties::PropertyType>(CSProperties::MATERIAL | CSProperties::METAL);
    const PrimitiveTypeCache& pecTypeCache = primitiveTypeCache(pecType);
    struct PrimitiveRange {
        unsigned int xStart, xStopEx, zStart, zStopEx;
    };
    std::vector<PrimitiveRange> primitiveRanges(pecTypeCache.prims.size());
    std::vector<PolygonRasterShape> primitiveShapes(pecTypeCache.prims.size());
    std::vector<std::uint8_t> canRasterize(pecTypeCache.prims.size(), 0);
    // A conformal prism decides only its Z edges here; applyConformalPEC() owns its X and Y edges.
    std::vector<std::uint8_t> conformalPrism(pecTypeCache.prims.size(), 0);
    for (std::size_t i = 0; i < pecTypeCache.prims.size(); ++i) {
        conformalPrism[i] = pecTypeCache.prims[i]->ToLinPoly() != nullptr &&
                            pecTypeCache.prims[i]->GetProperty()->ExistAttribute(kConformalAttribute);
        const double* bb = pecTypeCache.boundBoxes[i].box.data();
        bool bounded = true;
        for (int n = 0; n < 6; ++n) {
            bounded = bounded && std::isfinite(bb[n]);
        }
        const auto [xStart, xStopEx] =
            boundedRange(xLines, std::min(bb[0], bb[1]), std::max(bb[0], bb[1]), bounded, nx);
        const auto [zStart, zStopEx] =
            boundedRange(zLines, std::min(bb[4], bb[5]), std::max(bb[4], bb[5]), bounded, nz);
        primitiveRanges[i] = {xStart, xStopEx, zStart, zStopEx};
        canRasterize[i] = tryBuildPolygonRasterShape(pecTypeCache.prims[i], primitiveShapes[i]) ? 1U : 0U;
    }

    // winner[axis][x*nz+z] -- one row's worth of PaintPECColumn's own per-column winnerCache,
    // widened to cover every column in the row at once (row-major, not column-major, is what lets a
    // CSPrimPolygon/CSPrimLinPoly's coverage be rasterized once per row instead of tested point by
    // point -- see this file's own top comment). Reused across rows rather than reallocated.
    std::vector<CSPrimitives*> winner[3];
    for (int a = 0; a < 3; ++a) {
        winner[a].assign(static_cast<std::size_t>(nx) * nz, nullptr);
    }

    std::vector<std::uint8_t> insideMask(nx);
    PolygonRasterScratch rasterScratch;

    for (unsigned int y = 0; y < ny; ++y) {
        for (int a = 0; a < 3; ++a) {
            std::fill(winner[a].begin(), winner[a].end(), nullptr);
        }

        std::vector<std::size_t> vPrimIdx = primitivesBoundBoxIndices(-1, static_cast<int>(y), -1, pecType);

        for (auto it = vPrimIdx.rbegin(); it != vPrimIdx.rend(); ++it) {
            CSPrimitives* prim = pecTypeCache.prims[*it];
            const double* bb = pecTypeCache.boundBoxes[*it].box.data();
            const PrimitiveRange& range = primitiveRanges[*it];
            const unsigned int xStart = range.xStart;
            const unsigned int xStopEx = range.xStopEx;
            const unsigned int zStart = range.zStart;
            const unsigned int zStopEx = range.zStopEx;
            if (xStart >= xStopEx || zStart >= zStopEx) {
                continue;
            }

            if (canRasterize[*it] != 0) {
                const PolygonRasterShape& shape = primitiveShapes[*it];
                for (int axis = conformalPrism[*it] != 0 ? 2 : 0; axis < 3; ++axis) {
                    const double rowCoord = (axis == 1) ? yDual[y] : yPrimary[y];
                    const std::vector<double>& fullCols = (axis == 0) ? xDual : xPrimary;
                    const auto cols = std::span<const double>(fullCols).subspan(xStart, xStopEx - xStart);
                    const auto mask = std::span<std::uint8_t>(insideMask).first(cols.size());
                    rasterizePolygonRowBytes(shape, rowCoord, cols, mask, rasterScratch);

                    for (unsigned int z = zStart; z < zStopEx; ++z) {
                        const double zc = (axis == 2) ? zDual[z] : zPrimary[z];
                        if (zc < bb[4] || zc > bb[5]) {
                            continue; // IsInside()'s own elevation-axis bbox check, done separately
                                      // since the rasterizer above only ever resolves the in-plane
                                      // (x,y) decision -- see rasterizePolygonRow()'s own doc comment.
                        }
                        for (unsigned int xi = 0; xi < mask.size(); ++xi) {
                            if (mask[xi]) {
                                winner[axis][static_cast<std::size_t>(xStart + xi) * nz + z] = prim;
                            }
                        }
                    }
                }
            } else {
                double coord[3];
                for (unsigned int z = zStart; z < zStopEx; ++z) {
                    for (int axis = 0; axis < 3; ++axis) {
                        coord[1] = (axis == 1) ? yDual[y] : yPrimary[y];
                        coord[2] = (axis == 2) ? zDual[z] : zPrimary[z];
                        for (unsigned int x = xStart; x < xStopEx; ++x) {
                            coord[0] = (axis == 0) ? xDual[x] : xPrimary[x];
                            if (prim->IsInside(coord)) {
                                winner[axis][static_cast<std::size_t>(x) * nz + z] = prim;
                            }
                        }
                    }
                }
            }
        }

        for (unsigned int x = 0; x < nx; ++x) {
            for (unsigned int z = 0; z < nz; ++z) {
                for (int n = 0; n < 3; ++n) {
                    CSPrimitives* w = winner[n][static_cast<std::size_t>(x) * nz + z];
                    if (w != nullptr && w->GetProperty()->GetType() == CSProperties::METAL) {
                        const std::size_t i = index(x, y, z);
                        _grid.vv[n][i] = 0.0F;
                        _grid.vi[n][i] = 0.0F;
                    }
                }
            }
        }
    }
}

void CopperOperator::zeroOuterHLayer() {
    // The last current (H) line per axis lies outside the actual FDTD domain and can never be
    // iterated by the leapfrog update -- ported from the unconditional tail of
    // Operator::ApplyMagneticBC ("the last current lines are outside the FDTD domain and cannot be
    // iterated by the FDTD engine"), which runs regardless of PMC boundary settings (kiems never
    // requests a PMC boundary at all, but this specific zeroing is not conditional on that -- it's a
    // mesh-domain-shape fact, not a boundary-condition effect).
    const unsigned int nLines[3] = {numLines(0), numLines(1), numLines(2)};
    for (int n = 0; n < 3; ++n) {
        const int nP = (n + 1) % 3;
        const int nPP = (n + 2) % 3;
        unsigned int pos[3];
        pos[n] = nLines[n] - 1;
        for (pos[nP] = 0; pos[nP] < nLines[nP]; ++pos[nP]) {
            for (pos[nPP] = 0; pos[nPP] < nLines[nPP]; ++pos[nPP]) {
                const std::size_t i = index(pos[0], pos[1], pos[2]);
                _grid.ii[n][i] = 0.0F;
                _grid.iv[n][i] = 0.0F;
                _grid.ii[nP][i] = 0.0F;
                _grid.iv[nP][i] = 0.0F;
                _grid.ii[nPP][i] = 0.0F;
                _grid.iv[nPP][i] = 0.0F;
            }
        }
    }
}

// ---- CFL timestep -- ported from Operator::CalcTimestep_Var3. AdrOp::Shift/GetShiftedPos's own
// "reflect to cell" boundary mode (SetReflection2Cell(), used throughout CalcTimestep_Var1/Var3)
// reduces, for the +-1-only steps this formula ever uses, to: an out-of-range shift collapses back
// to the *original* (unshifted) index rather than wrapping or erroring -- algebraically verified
// against AdrOp::GetPos's own reflection formula (`muiIrel=-2*uiIpos-muiIrel-uiTypeOffset` etc, with
// uiTypeOffset=1 for cell reflection) rather than re-deriving AdrOp's own indexing class. ----

double CopperOperator::localTimestepVar3(int n, const unsigned int pos[3]) const {
    const unsigned int nx = numLines(0);
    const unsigned int ny = numLines(1);
    const unsigned int nz = numLines(2);
    const int nP = (n + 1) % 3;
    const int nPP = (n + 2) % 3;

    // Shifts `base` by `shift[axis]` per axis (each entry in {-1,0,1} only, ever), reflecting an
    // out-of-range result back to `base` itself on that axis -- see this section's own top comment.
    auto shiftedIndex = [&](const unsigned int base[3], const int shift[3]) -> std::size_t {
        unsigned int p[3];
        const unsigned int lineCount[3] = {nx, ny, nz};
        for (int i = 0; i < 3; ++i) {
            const long v = static_cast<long>(base[i]) + shift[i];
            p[i] = (v < 0 || v > static_cast<long>(lineCount[i]) - 1) ? base[i] : static_cast<unsigned int>(v);
        }
        return index(p[0], p[1], p[2]);
    };

    const std::size_t ipos0 = index(pos[0], pos[1], pos[2]);
    // An edge or face held inert (applyConformalPEC(): metal, so no unknown at all) is absent from
    // the system: an inert edge's own row sets no limit, and inert couplings drop out of the sums
    // and of the minimum the Var3 estimate subtracts -- left in as zeros, they would be taken as the
    // minimum and loosen the estimate's cancellation, as if removing an edge could cost timestep.
    if (_ecC[n][ipos0] >= kInertCoefficient) return std::numeric_limits<double>::infinity();
    int shift[3] = {0, 0, 0};
    int tmp[3];
    auto withTemp = [&](int axis, int step) {
        tmp[0] = shift[0];
        tmp[1] = shift[1];
        tmp[2] = shift[2];
        tmp[axis] += step;
        return shiftedIndex(pos, tmp);
    };

    double wqp = 1.0 / (_ecL[nPP][ipos0] * _ecC[n][withTemp(nP, 1)]) + 1.0 / (_ecL[nPP][ipos0] * _ecC[n][ipos0]);
    wqp += 1.0 / (_ecL[nP][ipos0] * _ecC[n][withTemp(nPP, 1)]) + 1.0 / (_ecL[nP][ipos0] * _ecC[n][ipos0]);

    shift[nP] = -1;
    const std::size_t iposA = shiftedIndex(pos, shift);
    wqp += 1.0 / (_ecL[nPP][iposA] * _ecC[n][withTemp(nP, 1)]) + 1.0 / (_ecL[nPP][iposA] * _ecC[n][iposA]);

    shift[nPP] = -1;
    const std::size_t iposB = shiftedIndex(pos, shift);
    wqp += 1.0 / (_ecL[nP][iposB] * _ecC[n][withTemp(nPP, 1)]) + 1.0 / (_ecL[nP][iposB] * _ecC[n][iposB]);

    // The sum of the present couplings less twice the smallest of them.
    const auto couplings = [](const std::array<std::pair<double, double>, 4>& lc) {
        double sum = 0.0;
        double smallest = std::numeric_limits<double>::infinity();
        for (const auto& [l, c] : lc) {
            if (l >= kInertCoefficient || c >= kInertCoefficient) continue;
            const double w = 1.0 / (l * c);
            sum += w;
            smallest = std::min(smallest, w);
        }
        return std::isfinite(smallest) ? sum - 2.0 * smallest : 0.0;
    };
    shift[0] = shift[1] = shift[2] = 0;
    const double wt1 = couplings({std::pair{_ecL[nPP][ipos0], _ecC[nP][ipos0]},
                                  std::pair{_ecL[nPP][withTemp(nP, -1)], _ecC[nP][ipos0]},
                                  std::pair{_ecL[nP][ipos0], _ecC[nPP][ipos0]},
                                  std::pair{_ecL[nP][withTemp(nPP, -1)], _ecC[nPP][ipos0]}});
    shift[0] = shift[1] = shift[2] = 0;
    const double wt2 = couplings({std::pair{_ecL[nPP][ipos0], _ecC[nP][withTemp(n, 1)]},
                                  std::pair{_ecL[nPP][withTemp(nP, -1)], _ecC[nP][withTemp(n, 1)]},
                                  std::pair{_ecL[nP][ipos0], _ecC[nPP][withTemp(n, 1)]},
                                  std::pair{_ecL[nP][withTemp(nPP, -1)], _ecC[nPP][withTemp(n, 1)]}});

    const double newT = 2.0 / std::sqrt(wqp + wt1 + wt2);
    return newT > 0.0 ? newT : std::numeric_limits<double>::infinity();
}

double CopperOperator::calcTimestepVar3() const {
    double dT = 1e200;
    unsigned int pos[3];
    for (int n = 0; n < 3; ++n) {
        for (pos[2] = 0; pos[2] < numLines(2); ++pos[2]) {
            for (pos[1] = 0; pos[1] < numLines(1); ++pos[1]) {
                for (pos[0] = 0; pos[0] < numLines(0); ++pos[0]) {
                    dT = std::min(dT, localTimestepVar3(n, pos));
                }
            }
        }
    }
    return dT;
}

double CopperOperator::largestEigenvalueInBox(const std::array<unsigned int, 3>& lo,
                                              const std::array<unsigned int, 3>& hi) const {
    const std::array<unsigned int, 3> size = {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1};
    const std::size_t count = static_cast<std::size_t>(size[0]) * size[1] * size[2];
    const auto local = [&](unsigned int x, unsigned int y, unsigned int z) {
        return (static_cast<std::size_t>(z) * size[1] + y) * size[0] + x;
    };
    // An inert (metal) edge or face, or one with no coefficient at all, has no unknown: weight 0.
    const auto inverse = [](double value) { return value > 0.0 && value < kInertCoefficient ? 1.0 / value : 0.0; };

    // Edge n at node q exists when q + e_n is in the box; face m at node f when f + e_mP and
    // f + e_mPP are. Everything else stays zero, so the box is a principal submatrix of the whole
    // operator and its lambda_max a lower bound -- the margin makes it a close one.
    std::array<std::vector<double>, 3> rootInvC, invL, x, y, faces;
    for (int n = 0; n < 3; ++n) {
        rootInvC[n].assign(count, 0.0);
        invL[n].assign(count, 0.0);
        x[n].assign(count, 0.0);
        y[n].assign(count, 0.0);
        faces[n].assign(count, 0.0);
    }
    for (unsigned int z = 0; z < size[2]; ++z) {
        for (unsigned int yy = 0; yy < size[1]; ++yy) {
            for (unsigned int xx = 0; xx < size[0]; ++xx) {
                const unsigned int q[3] = {xx, yy, z};
                const std::size_t l = local(xx, yy, z);
                const std::size_t g = index(lo[0] + xx, lo[1] + yy, lo[2] + z);
                for (int n = 0; n < 3; ++n) {
                    const int nP = (n + 1) % 3, nPP = (n + 2) % 3;
                    if (q[n] + 1 < size[n]) rootInvC[n][l] = std::sqrt(inverse(_ecC[n][g]));
                    if (q[nP] + 1 < size[nP] && q[nPP] + 1 < size[nPP]) invL[n][l] = inverse(_ecL[n][g]);
                    // A fixed pseudo-random start, so no mode is missed by symmetry.
                    const std::uint64_t h = (static_cast<std::uint64_t>(l) * 3 + n + 1) * 0x9E3779B97F4A7C15ULL;
                    x[n][l] = rootInvC[n][l] > 0.0 ? 0.5 + static_cast<double>(h >> 11) / 9007199254740992.0 : 0.0;
                }
            }
        }
    }

    // A = S D^T L^-1 D S with S = C^-1/2 is symmetric, with the same eigenvalues as C^-1 D^T L^-1 D;
    // its Rayleigh quotient rises to lambda_max under power iteration.
    const auto step = [&](const std::array<std::vector<double>, 3>& in, std::array<std::vector<double>, 3>& out) {
        for (int m = 0; m < 3; ++m) {
            const int mP = (m + 1) % 3, mPP = (m + 2) % 3;
            for (unsigned int z = 0; z < size[2]; ++z) {
                for (unsigned int yy = 0; yy < size[1]; ++yy) {
                    for (unsigned int xx = 0; xx < size[0]; ++xx) {
                        const std::size_t f = local(xx, yy, z);
                        if (invL[m][f] == 0.0) {
                            faces[m][f] = 0.0;
                            continue;
                        }
                        unsigned int p[3] = {xx, yy, z};
                        ++p[mP];
                        const std::size_t fP = local(p[0], p[1], p[2]);
                        --p[mP];
                        ++p[mPP];
                        const std::size_t fPP = local(p[0], p[1], p[2]);
                        const double curl = in[mPP][fP] * rootInvC[mPP][fP] - in[mPP][f] * rootInvC[mPP][f] -
                                            in[mP][fPP] * rootInvC[mP][fPP] + in[mP][f] * rootInvC[mP][f];
                        faces[m][f] = invL[m][f] * curl;
                    }
                }
            }
        }
        for (int n = 0; n < 3; ++n) {
            const int nP = (n + 1) % 3, nPP = (n + 2) % 3;
            for (unsigned int z = 0; z < size[2]; ++z) {
                for (unsigned int yy = 0; yy < size[1]; ++yy) {
                    for (unsigned int xx = 0; xx < size[0]; ++xx) {
                        const std::size_t e = local(xx, yy, z);
                        if (rootInvC[n][e] == 0.0) {
                            out[n][e] = 0.0;
                            continue;
                        }
                        unsigned int p[3] = {xx, yy, z};
                        double sum = faces[nPP][e] - faces[nP][e];
                        if (p[nP] > 0) {
                            --p[nP];
                            sum -= faces[nPP][local(p[0], p[1], p[2])];
                            ++p[nP];
                        }
                        if (p[nPP] > 0) {
                            --p[nPP];
                            sum += faces[nP][local(p[0], p[1], p[2])];
                        }
                        out[n][e] = rootInvC[n][e] * sum;
                    }
                }
            }
        }
    };
    const auto dot = [&](const std::array<std::vector<double>, 3>& a, const std::array<std::vector<double>, 3>& b) {
        double sum = 0.0;
        for (int n = 0; n < 3; ++n) {
            for (std::size_t i = 0; i < count; ++i) sum += a[n][i] * b[n][i];
        }
        return sum;
    };

    // Lanczos: the largest Ritz value of the tridiagonal T it builds rises to lambda_max far faster
    // than a power iteration's Rayleigh quotient -- which matters most exactly when a conformal mode
    // only just clears the bulk spectrum. Ritz values never exceed lambda_max, and only the largest
    // is wanted, so the lost orthogonality of plain Lanczos (duplicate Ritz values) does no harm.
    const double initialNorm = std::sqrt(dot(x, x));
    if (initialNorm == 0.0) return 0.0;
    for (int n = 0; n < 3; ++n) {
        for (double& v : x[n]) v /= initialNorm;
    }
    std::array<std::vector<double>, 3> previous;
    for (int n = 0; n < 3; ++n) previous[n].assign(count, 0.0);
    std::vector<double> alpha, beta; // T's diagonal, and its off-diagonal (beta[j] couples j, j + 1)
    const auto largestRitzValue = [&]() {
        // Bisection on the Sturm count: the number of T's eigenvalues below `at`.
        const std::size_t k = alpha.size();
        const auto below = [&](double at) {
            std::size_t count = 0;
            double d = 1.0;
            for (std::size_t i = 0; i < k; ++i) {
                const double b2 = i > 0 ? beta[i - 1] * beta[i - 1] : 0.0;
                d = alpha[i] - at - (i > 0 ? b2 / d : 0.0);
                if (d == 0.0) d = -1e-300;
                if (d < 0.0) ++count;
            }
            return count;
        };
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (std::size_t i = 0; i < k; ++i) {
            const double radius = (i > 0 ? std::fabs(beta[i - 1]) : 0.0) + (i + 1 < k ? std::fabs(beta[i]) : 0.0);
            lo = std::min(lo, alpha[i] - radius);
            hi = std::max(hi, alpha[i] + radius);
        }
        for (int step = 0; step < 200 && hi - lo > 1e-12 * std::fabs(hi); ++step) {
            const double mid = (lo + hi) / 2;
            (below(mid) < k ? lo : hi) = mid;
        }
        return hi;
    };

    constexpr int kMaxIterations = 400;
    constexpr int kCheckEvery = 10;
    double lambda = 0.0;
    double lastChecked = 0.0;
    for (int iteration = 1; iteration <= kMaxIterations; ++iteration) {
        step(x, y);
        const double a = dot(x, y);
        const double b = beta.empty() ? 0.0 : beta.back();
        for (int n = 0; n < 3; ++n) {
            for (std::size_t i = 0; i < count; ++i) y[n][i] -= a * x[n][i] + b * previous[n][i];
        }
        alpha.push_back(a);
        const double next = std::sqrt(dot(y, y));
        const bool exhausted = next <= 1e-12 * std::fabs(a);
        if (iteration % kCheckEvery == 0 || exhausted || iteration == kMaxIterations) {
            lambda = largestRitzValue();
            if (exhausted || lambda - lastChecked <= 1e-9 * lambda) break;
            lastChecked = lambda;
        }
        beta.push_back(next);
        std::swap(previous, x);
        for (int n = 0; n < 3; ++n) {
            for (std::size_t i = 0; i < count; ++i) x[n][i] = y[n][i] / next;
        }
    }
    return lambda;
}

double CopperOperator::conformalStableTimestep() const {
    const CSProperties::PropertyType pecType =
        static_cast<CSProperties::PropertyType>(CSProperties::MATERIAL | CSProperties::METAL);
    const PrimitiveTypeCache& cache = primitiveTypeCache(pecType);
    std::set<std::array<unsigned int, 6>> boxes;
    for (std::size_t p = 0; p < cache.prims.size(); ++p) {
        const CSProperties* property = cache.prims[p]->GetProperty();
        if (!property->ExistAttribute(kConformalAttribute) && !property->ExistAttribute(kThinWireRadiusAttribute)) continue;
        if (!cache.boundBoxes[p].ok) continue;
        const auto& bb = cache.boundBoxes[p].box;
        std::array<unsigned int, 6> box{};
        for (int axis = 0; axis < 3; ++axis) {
            const std::vector<double>& lines = _rawLines[axis];
            const auto lineCount = static_cast<unsigned int>(lines.size());
            const double a = std::min(bb[2 * axis], bb[2 * axis + 1]), b = std::max(bb[2 * axis], bb[2 * axis + 1]);
            auto first = static_cast<unsigned int>(std::lower_bound(lines.begin(), lines.end(), a) - lines.begin());
            auto last = static_cast<unsigned int>(std::upper_bound(lines.begin(), lines.end(), b) - lines.begin());
            first = first > kConformalStabilityMarginLines ? first - kConformalStabilityMarginLines : 0;
            last = std::min(last + kConformalStabilityMarginLines, lineCount - 1);
            box[2 * axis] = first;
            box[2 * axis + 1] = std::max(first, last);
        }
        boxes.insert(box);
    }
    double lambda = 0.0;
    for (const auto& box : boxes) {
        lambda = std::max(lambda, largestEigenvalueInBox({box[0], box[2], box[4]}, {box[1], box[3], box[5]}));
    }
    return lambda > 0.0 ? 2.0 / std::sqrt(lambda) : std::numeric_limits<double>::infinity();
}

// ---- conformal metal and thin wires -- see CopperConformal.hpp and applyConformalPEC()'s own doc
// comment ----

bool CopperOperator::hasConformalGeometry() const {
    const CSProperties::PropertyType pecType =
        static_cast<CSProperties::PropertyType>(CSProperties::MATERIAL | CSProperties::METAL);
    for (CSPrimitives* prim : primitiveTypeCache(pecType).prims) {
        const CSProperties* property = prim->GetProperty();
        if (property->ExistAttribute(kConformalAttribute) || property->ExistAttribute(kThinWireRadiusAttribute)) {
            return true;
        }
    }
    return false;
}

void CopperOperator::applyConformalPEC(double baseTimestep) {
    // Fractions are snapped to 1/128ths: finer than any geometry the mesh resolves, and coarse
    // enough that the coefficient table (CopperCoefficientTable) keeps few distinct entries.
    constexpr double kQuantum = 1.0 / 128.0;
    const auto quantise = [](double fraction) { return std::clamp(std::round(fraction / kQuantum) * kQuantum, 0.0, 1.0); };
    // An edge or face with no dynamics, for the timestep derivation only -- see localTimestepVar3().
    const auto holdInert = [&](bool electric, int axis, std::size_t i) {
        std::vector<double>& values = electric ? _ecC[axis] : _ecL[axis];
        if (values[i] >= kInertCoefficient) return;
        _inertCoefficients.push_back({electric, axis, i, values[i]});
        values[i] = kInertCoefficient;
    };

    const auto& X = _rawLines[0];
    const auto& Y = _rawLines[1];
    const auto& Z = _rawLines[2];
    const unsigned int nx = numLines(0), ny = numLines(1), nz = numLines(2);
    const auto lineRange = [](const std::vector<double>& lines, double lo, double hi) {
        auto first = static_cast<unsigned int>(std::lower_bound(lines.begin(), lines.end(), lo) - lines.begin());
        auto last = static_cast<unsigned int>(std::upper_bound(lines.begin(), lines.end(), hi) - lines.begin());
        if (first > 0) --first;
        return std::pair{first, std::min<unsigned int>(last + 1, static_cast<unsigned int>(lines.size()))};
    };

    const CSProperties::PropertyType pecType =
        static_cast<CSProperties::PropertyType>(CSProperties::MATERIAL | CSProperties::METAL);
    const PrimitiveTypeCache& cache = primitiveTypeCache(pecType);
    std::vector<std::array<unsigned int, 4>> faces; // axis, x, y, z of every partly open face
    for (std::size_t p = 0; p < cache.prims.size(); ++p) {
        CSPrimitives* prim = cache.prims[p];
        if (!prim->GetProperty()->ExistAttribute(kConformalAttribute) || !cache.boundBoxes[p].ok) continue;
        PolygonRasterShape shape;
        if (!tryBuildPolygonRasterShape(prim, shape) || prim->ToLinPoly() == nullptr) continue;
        const conformal::Polygon polygon{shape.x, shape.y};
        const auto& bb = cache.boundBoxes[p].box;
        const double zLo = std::min(bb[4], bb[5]), zHi = std::max(bb[4], bb[5]);
        const auto [x0, x1] = lineRange(X, std::min(bb[0], bb[1]), std::max(bb[0], bb[1]));
        const auto [y0, y1] = lineRange(Y, std::min(bb[2], bb[3]), std::max(bb[2], bb[3]));
        const auto [z0, z1] = lineRange(Z, zLo, zHi);

        // Z edges at nodes inside: metal, decided by computePEC() exactly as for any prism.
        std::vector<std::uint8_t> inside(x1 - x0);
        PolygonRasterScratch scratch;
        for (unsigned int y = y0; y < y1; ++y) {
            rasterizePolygonRowBytes(shape, Y[y], std::span<const double>(X).subspan(x0, x1 - x0), inside, scratch);
            for (unsigned int z = z0; z + 1 < nz && z < z1; ++z) {
                const double zMid = (Z[z] + Z[z + 1]) / 2;
                if (zMid < zLo || zMid > zHi) continue;
                for (unsigned int x = x0; x < x1; ++x) {
                    if (inside[x - x0]) holdInert(true, 2, index(x, y, z));
                }
            }
        }

        for (unsigned int z = z0; z < z1; ++z) {
            const bool onLine = Z[z] >= zLo && Z[z] <= zHi;
            const double cellOverlap =
                z + 1 < nz ? std::max(0.0, std::min(Z[z + 1], zHi) - std::max(Z[z], zLo)) / (Z[z + 1] - Z[z]) : 0.0;
            for (unsigned int y = y0; y < y1; ++y) {
                for (unsigned int x = x0; x < x1; ++x) {
                    const std::size_t i = index(x, y, z);
                    const double dx = x + 1 < nx ? X[x + 1] - X[x] : 0.0;
                    const double dy = y + 1 < ny ? Y[y + 1] - Y[y] : 0.0;
                    const double insideX = dx > 0 ? conformal::insideLength(polygon, true, Y[y], X[x], X[x + 1]) / dx : 0.0;
                    const double insideY = dy > 0 ? conformal::insideLength(polygon, false, X[x], Y[y], Y[y + 1]) / dy : 0.0;
                    if (onLine) {
                        // X and Y edges on this plane carry only their open length.
                        for (const auto& [axis, covered] : {std::pair{0, insideX}, std::pair{1, insideY}}) {
                            if (covered <= 0.0) continue;
                            const double open = quantise(1.0 - covered);
                            if (open <= 0.0) {
                                _conformalMetalEdges.emplace_back(axis, i);
                                holdInert(true, axis, i);
                            } else if (open < 1.0) {
                                _ecC[axis][i] /= open;
                                _ecG[axis][i] /= open;
                            }
                        }
                        // The Z face on this plane carries flux through its open area only.
                        if (dx > 0 && dy > 0) {
                            const double covered = conformal::insideArea(polygon, X[x], X[x + 1], Y[y], Y[y + 1]) / (dx * dy);
                            if (covered > 0.0) faces.push_back({2, x, y, z});
                        }
                    }
                    // X and Y faces between this plane and the next, by the share of their edge
                    // inside the cross-section times their share of the prism's height.
                    if (cellOverlap > 0.0) {
                        if (insideY > 0.0) faces.push_back({0, x, y, z});
                        if (insideX > 0.0) faces.push_back({1, x, y, z});
                    }
                }
            }
        }

        // Each cut face carries flux through its open area only, unless that would cost more
        // timestep than kConformalStability allows -- see below.
        for (const auto& [axis, x, y, z] : faces) {
            const std::size_t i = index(x, y, z);
            double covered = 0.0;
            if (axis == 2) {
                covered = conformal::insideArea(polygon, X[x], X[x + 1], Y[y], Y[y + 1]) / ((X[x + 1] - X[x]) * (Y[y + 1] - Y[y]));
            } else {
                const double overlap = std::max(0.0, std::min(Z[z + 1], zHi) - std::max(Z[z], zLo)) / (Z[z + 1] - Z[z]);
                covered = overlap * (axis == 0 ? conformal::insideLength(polygon, false, X[x], Y[y], Y[y + 1]) / (Y[y + 1] - Y[y])
                                               : conformal::insideLength(polygon, true, Y[y], X[x], X[x + 1]) / (X[x + 1] - X[x]));
            }
            const double open = quantise(1.0 - covered);
            if (open >= 1.0 || _ecL[axis][i] >= kInertCoefficient) continue;
            if (open <= 0.0) {
                // Wholly inside, bounded by metal edges: no dynamics.
                _conformalClosedFaces.emplace_back(axis, i);
                holdInert(false, axis, i);
                continue;
            }
            // The flux through the open area alone raises this face's coefficient by 1/open. Where that
            // would cost more timestep than allowed, the face gives up a little geometric precision
            // instead, as Benkler et al. trade it: whichever misplaces less of it, either it keeps the
            // smallest open fraction that is stable, or (a sliver) it joins the metal.
            const double inductance = _ecL[axis][i];
            const double resistance = _ecR[axis][i];
            const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
            const unsigned int lines[3] = {nx, ny, nz};
            const unsigned int face[3] = {x, y, z};
            // localTimestepVar3(n, pos) reads the face along nPP at pos and pos - e(nP), and along nP
            // at pos, pos - e(nPP) and pos - e(nP) - e(nPP).
            struct Term {
                int n;
                int shiftA, shiftB; // -1 for none
            };
            const Term terms[5] = {{a1, -1, -1}, {a1, a2, -1}, {a2, -1, -1}, {a2, a1, -1}, {a2, a1, static_cast<int>(axis)}};
            const auto stableWith = [&](double fraction) {
                _ecL[axis][i] = inductance * fraction;
                for (const Term& term : terms) {
                    unsigned int pos[3] = {face[0], face[1], face[2]};
                    bool inside = true;
                    for (const int shift : {term.shiftA, term.shiftB}) {
                        if (shift < 0) continue;
                        if (pos[shift] + 1 >= lines[shift]) inside = false;
                        else ++pos[shift];
                    }
                    if (inside && localTimestepVar3(term.n, pos) < kConformalStability * baseTimestep) return false;
                }
                return true;
            };
            double fraction = open;
            if (!stableWith(fraction)) {
                double lo = open, hi = 1.0;
                for (int step = 0; step < 12; ++step) {
                    const double mid = (lo + hi) / 2;
                    (stableWith(mid) ? hi : lo) = mid;
                }
                fraction = quantise(hi + kQuantum / 2);
                if (open < fraction - open) {
                    // The face closes and every edge around it becomes metal too: an open edge left
                    // facing a closed face would act as a magnetic wall and choke the current around
                    // the conductor. The surface moves out here by the sliver, as Dey and Mittra
                    // treat their smallest cells.
                    _ecL[axis][i] = inductance;
                    _conformalClosedFaces.emplace_back(axis, i);
                    holdInert(false, axis, i);
                    const int p1 = (axis + 1) % 3, p2 = (axis + 2) % 3;
                    for (const auto& [edgeAxis, along] : {std::pair{p1, p2}, std::pair{p2, p1}}) {
                        for (const unsigned int step : {0U, 1U}) {
                            unsigned int pos[3] = {face[0], face[1], face[2]};
                            pos[along] += step;
                            if (pos[along] >= lines[along]) continue;
                            const std::size_t edge = index(pos[0], pos[1], pos[2]);
                            _conformalMetalEdges.emplace_back(edgeAxis, edge);
                            holdInert(true, edgeAxis, edge);
                        }
                    }
                    continue;
                }
            }
            _ecL[axis][i] = inductance * fraction;
            _ecR[axis][i] = resistance * fraction;
        }
        faces.clear();
    }

    // Thin wires: Railton's modified parameters on the four radial edges and the faces circling the
    // wire, along every node of the box's line.
    for (std::size_t p = 0; p < cache.prims.size(); ++p) {
        CSPrimitives* prim = cache.prims[p];
        const CSProperties* property = prim->GetProperty();
        if (!property->ExistAttribute(kThinWireRadiusAttribute) || !cache.boundBoxes[p].ok) continue;
        const double radius = std::atof(property->GetAttributeValue(kThinWireRadiusAttribute).c_str());
        const auto& bb = cache.boundBoxes[p].box;
        const double centre[3] = {(bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2, (bb[4] + bb[5]) / 2};
        unsigned int node[3];
        snapToMesh(centre, node, false);
        const unsigned int x = node[0], y = node[1];
        if (!(radius > 0.0) || x == 0 || y == 0 || x + 1 >= nx || y + 1 >= ny) continue;
        const double zLo = std::min(bb[4], bb[5]), zHi = std::max(bb[4], bb[5]);
        const double acrossX = (X[x + 1] - X[x - 1]) / 2, acrossY = (Y[y + 1] - Y[y - 1]) / 2;
        // Each radial edge: its axis, the position of its own index, its length, and the face beside it.
        struct Radial {
            int axis;
            unsigned int ex, ey;
            double along, across;
        };
        const Radial radials[4] = {{0, x, y, X[x + 1] - X[x], acrossY},
                                   {0, x - 1, y, X[x] - X[x - 1], acrossY},
                                   {1, x, y, Y[y + 1] - Y[y], acrossX},
                                   {1, x, y - 1, Y[y] - Y[y - 1], acrossX}};
        for (unsigned int z = 0; z < nz; ++z) {
            const bool onLine = Z[z] >= zLo && Z[z] <= zHi;
            const bool cellInside = z + 1 < nz && Z[z] >= zLo && Z[z + 1] <= zHi;
            for (const Radial& radial : radials) {
                if (radius >= 0.9 * radial.along) continue; // the wire fills its cell: no thin-wire regime
                const double mu = conformal::thinWirePermeabilityFactor(radius, radial.along, radial.across);
                const std::size_t i = index(radial.ex, radial.ey, z);
                if (onLine) {
                    _ecC[radial.axis][i] /= mu;
                    _ecG[radial.axis][i] /= mu;
                }
                // The H component circling the wire beside a radial X edge is Hy, beside a Y edge Hx.
                if (cellInside) {
                    const int circling = radial.axis == 0 ? 1 : 0;
                    _ecL[circling][i] *= mu;
                    _ecR[circling][i] *= mu;
                }
            }
        }
    }
}

void CopperOperator::restoreInertCoefficients() {
    for (const InertCoefficient& held : _inertCoefficients) {
        (held.electric ? _ecC : _ecL)[held.axis][held.index] = held.value;
    }
    _inertCoefficients.clear();
}

void CopperOperator::zeroConformalPEC() {
    for (const auto& [axis, i] : _conformalMetalEdges) {
        _grid.vv[axis][i] = 0.0F;
        _grid.vi[axis][i] = 0.0F;
    }
    for (const auto& [axis, i] : _conformalClosedFaces) {
        _grid.ii[axis][i] = 0.0F;
        _grid.iv[axis][i] = 0.0F;
    }
}

// ---- excitation -- ported from Excitation::CalcGaussianPulsExcitation +
// Operator_Ext_Excitation::BuildExtension's soft-E-field (ExcitType 0) branch only ----

void CopperOperator::computeExcitation() {
    const double dT = _grid.timestepSeconds;
    _excitation.signalPeriodSeconds = 0.0; // Gaussian pulse is one-shot, never periodic

    if (dT <= 0.0 || _config.fc <= 0.0) {
        return;
    }

    // -- CalcGaussianPulsExcitation --
    auto length = static_cast<unsigned int>(std::ceil(2.0 * 9.0 / (2.0 * physical::pi * _config.fc) / dT));
    if (length > _config.maxTimesteps) {
        length = _config.maxTimesteps;
    }
    if (length == 0) {
        return;
    }
    _excitation.voltageSignal.assign(length, 0.0F);
    _excitation.currentSignal.assign(length, 0.0F);
    for (unsigned int n = 1; n < length; ++n) {
        double t = static_cast<double>(n) * dT;
        _excitation.voltageSignal[n] = static_cast<float>(
            std::cos(2.0 * physical::pi * _config.f0 *
                     (t - 9.0 / (2.0 * physical::pi * _config.fc))) *
            std::exp(-1.0 * std::pow(2.0 * physical::pi * _config.fc * t / 3.0 - 3.0, 2)));
        t += 0.5 * dT;
        _excitation.currentSignal[n] = static_cast<float>(
            std::cos(2.0 * physical::pi * _config.f0 *
                     (t - 9.0 / (2.0 * physical::pi * _config.fc))) *
            std::exp(-1.0 * std::pow(2.0 * physical::pi * _config.fc * t / 3.0 - 3.0, 2)));
    }

    // -- Operator_Ext_Excitation::BuildExtension, soft-E-field (ExcitType 0) branch only --
    std::vector<CSProperties*> excProps = _csx.GetPropertyByType(CSProperties::EXCITATION);
    if (excProps.empty()) {
        return;
    }

    const unsigned int nx = numLines(0);
    const unsigned int ny = numLines(1);
    const unsigned int nz = numLines(2);
    unsigned int pos[3];
    double coord[3];
    for (pos[2] = 0; pos[2] < nz; ++pos[2]) {
        for (pos[1] = 0; pos[1] < ny; ++pos[1]) {
            std::vector<CSPrimitives*> vPrims =
                primitivesBoundBox(-1, static_cast<int>(pos[1]), static_cast<int>(pos[2]), CSProperties::EXCITATION);

            for (pos[0] = 0; pos[0] < nx; ++pos[0]) {
                for (int n = 0; n < 3; ++n) {
                    if (!yeeCoords(n, pos, coord, false)) {
                        continue;
                    }
                    CSProperties* prop = _csx.GetPropertyByCoordPriority(coord, vPrims, true);
                    if (prop == nullptr) {
                        continue;
                    }
                    CSPropExcitation* elec = prop->ToExcitation();
                    if (elec == nullptr || !elec->GetEnabled() || elec->GetExcitType() != 0 || !elec->GetActiveDir(n)) {
                        continue;
                    }
                    const double amp = elec->GetWeightedExcitation(n, coord) * edgeLength(n, pos, false);
                    if (amp == 0.0) {
                        continue;
                    }
                    CopperExcitationCell cell;
                    cell.x = pos[0];
                    cell.y = pos[1];
                    cell.z = pos[2];
                    cell.axis = static_cast<std::uint32_t>(n);
                    cell.amplitude = static_cast<float>(amp);
                    cell.delaySteps = static_cast<std::uint32_t>(elec->GetDelay() / dT);
                    _excitation.voltageCells.push_back(cell);
                }
            }
        }
    }
}

// ---- lumped elements (PARALLEL only) -- ported from Operator::Calc_LumpedElements' parallel-RC
// path (operator.cpp), which folds directly into EC_C/EC_G then rederives vv/vi for just the
// covered edges. SERIES is handled entirely separately, at runtime, by CopperLumpedRLC.hpp/
// CopperFDTDRunner.cpp -- see that header's own top comment. ----

void CopperOperator::computeParallelLumpedElements() {
    std::vector<CSProperties*> lumpedProps = _csx.GetPropertyByType(CSProperties::LUMPED_ELEMENT);
    const double dT = _grid.timestepSeconds;
    for (CSProperties* prop : lumpedProps) {
        auto* lumped = dynamic_cast<CSPropLumpedElement*>(prop);
        if (lumped == nullptr || lumped->GetLEtype() != CSPropLumpedElement::PARALLEL) {
            continue;
        }
        const int dir = lumped->GetDirection();
        if (dir < 0 || dir > 2) {
            continue;
        }
        const int dirP1 = (dir + 1) % 3;
        const int dirP2 = (dir + 2) % 3;

        const double rawR = lumped->GetResistance();
        const double rawC = lumped->GetCapacity();
        const double R = (std::isnan(rawR) || rawR < 0.0) ? -1.0 : rawR; // -1 sentinel matches "R absent" (NaN) below
        const bool hasC = !std::isnan(rawC) && rawC > 0.0;
        const double C = hasC ? rawC : 0.0;
        for (std::size_t p = 0; p < lumped->GetQtyPrimitives(); ++p) {
            auto* box = dynamic_cast<CSPrimBox*>(lumped->GetPrimitive(p));
            if (box == nullptr) {
                continue;
            }
            double dstart[3], dstop[3];
            for (int n = 0; n < 3; ++n) {
                dstart[n] = box->GetCoord(2 * n);
                dstop[n] = box->GetCoord(2 * n + 1);
            }
            unsigned int uiStart[3], uiStop[3];
            const int snapDim = snapBox2Mesh(dstart, dstop, uiStart, uiStop, /*dualMesh=*/false, /*snapMethod=*/0);
            if (snapDim <= 0 || uiStart[dir] == uiStop[dir]) {
                continue;
            }

            // Transverse edges are parallel branches and longitudinal edges are in series. Weight
            // them using their real A/l geometry, exactly like Operator::Calc_LumpedElements(). A
            // cell-count-only split is valid only on a uniform cubic mesh; on the strongly
            // non-uniform board mesh it made a nominal 45-ohm termination depend on local cell
            // dimensions and could weaken it by orders of magnitude.
            unsigned int pos[3] = {0, 0, 0};
            double inverseUnitGC = 0.0;
            for (pos[dir] = uiStart[dir]; pos[dir] < uiStop[dir]; ++pos[dir]) {
                double unitGCPlane = 0.0;
                for (pos[dirP1] = uiStart[dirP1]; pos[dirP1] <= uiStop[dirP1]; ++pos[dirP1]) {
                    for (pos[dirP2] = uiStart[dirP2]; pos[dirP2] <= uiStop[dirP2]; ++pos[dirP2]) {
                        const double length = edgeLength(dir, pos);
                        if (length > 0.0) unitGCPlane += edgeArea(dir, pos) / length;
                    }
                }
                if (unitGCPlane > 0.0) inverseUnitGC += 1.0 / unitGCPlane;
            }
            if (inverseUnitGC <= 0.0) continue;
            const double unitGC = 1.0 / inverseUnitGC;
            const double conductivity = R > 0.0 ? 1.0 / (R * unitGC) : 0.0;
            const double permittivity = hasC ? C / unitGC : 0.0;

            for (pos[dir] = uiStart[dir]; pos[dir] < uiStop[dir]; ++pos[dir]) {
                for (pos[dirP1] = uiStart[dirP1]; pos[dirP1] <= uiStop[dirP1]; ++pos[dirP1]) {
                    for (pos[dirP2] = uiStart[dirP2]; pos[dirP2] <= uiStop[dirP2]; ++pos[dirP2]) {
                        const std::size_t i = index(pos[0], pos[1], pos[2]);
                        const double length = edgeLength(dir, pos);
                        if (length <= 0.0) continue;
                        const double geometricFactor = edgeArea(dir, pos) / length;
                        const double dG = conductivity * geometricFactor;
                        const double dC = permittivity * geometricFactor;
                        if (dC > 0.0) {
                            _ecC[dir][i] = dC;
                        }
                        // else: keep the natural grid capacitance already computed by
                        // computeMaterialCoefficients() -- the real extension additionally bumps a
                        // too-small natural capacitance upward for ADE stability
                        // (Zmin/Zcd_min/LUMPED_RLC_Z_FACT); not reproduced here since it only
                        // matters for a parallel inductance, which libkiems does not create.

                        if (R >= 0.0) {
                            _ecG[dir][i] = dG;
                        }

                        // vv[dir]/vi[dir] rederived for just this edge from the now-mutated EC_C/EC_G
                        // (EC_L/EC_R untouched) -- mirrors Operator_Ext_LumpedRLC's own immediate
                        // m_Op->Calc_ECOperatorPos(dir,pos) call.
                        const double c = _ecC[dir][i];
                        const double g = _ecG[dir][i];
                        if (c > 0.0) {
                            _grid.vv[dir][i] = static_cast<float>((1.0 - dT * g / 2.0 / c) / (1.0 + dT * g / 2.0 / c));
                            _grid.vi[dir][i] = static_cast<float>((dT / c) / (1.0 + dT * g / 2.0 / c));
                        } else {
                            _grid.vv[dir][i] = 0.0F;
                            _grid.vi[dir][i] = 0.0F;
                        }
                    }
                }
            }

            // LumpedPort requests equipotential metal end caps. These are part of openEMS's
            // parallel-lumped-element semantics and ensure the resistor face really connects to
            // the zero-thickness pad and reference plane across its whole footprint.
            if (lumped->GetCaps()) {
                for (pos[dirP1] = uiStart[dirP1]; pos[dirP1] <= uiStop[dirP1]; ++pos[dirP1]) {
                    for (pos[dirP2] = uiStart[dirP2]; pos[dirP2] <= uiStop[dirP2]; ++pos[dirP2]) {
                        for (const unsigned int end : {uiStart[dir], uiStop[dir]}) {
                            pos[dir] = end;
                            const std::size_t i = index(pos[0], pos[1], pos[2]);
                            if (pos[dirP1] < uiStop[dirP1]) {
                                _grid.vv[dirP1][i] = 0.0F;
                                _grid.vi[dirP1][i] = 0.0F;
                            }
                            if (pos[dirP2] < uiStop[dirP2]) {
                                _grid.vv[dirP2][i] = 0.0F;
                                _grid.vi[dirP2][i] = 0.0F;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---- construction ----

CopperOperator::CopperOperator(ContinuousStructure& csx, const Config& config) : _csx(csx), _config(config) {
    // openEMS's own Operator::SetGeometryCSX() is only ever reached after openEMS::SetupFDTD() has
    // already called ContinuousStructure::Update() on the whole scene graph -- this constructor is
    // this class's own equivalent entry point, so it must do the same, unconditionally, rather than
    // assume some earlier caller already did. Update() itself is nearly a no-op for every primitive/
    // property whose values are plain doubles (which is everything kiems ever builds -- no string-
    // formula parametrics) -- ParameterScalar::Evaluate() returns immediately whenever ParameterMode
    // is false -- *except* for CSPrimPolygon/CSPrimLinPoly (kiems's own copper trace/pad/plane
    // representation), whose Update() also computes and caches m_BoundBox, which IsInside() then
    // uses as a mandatory quick-reject check before its real point-in-polygon test (see
    // CSPrimPolygon::IsInside()'s own "m_BoundBox[2*n]>Coord[n]" checks -- with m_BoundBoxValid still
    // at its constructed-default false/zeroed state, that check rejects every real-world coordinate
    // unconditionally, making every polygon primitive invisible to computeMaterialCoefficients()/
    // computePEC() below). Confirmed as a real regression this way: skipping the caller's own
    // SetupFDTD() call (once real openEMS's Operator/Engine were no longer needed at all for a CPML
    // run) made every copper trace/plane vanish from the painted geometry -- only CSPrimBox-shaped
    // primitives (whose own IsInside() reads m_Coords directly, unaffected) kept working, which is
    // exactly why a port's own excitation box still worked while the trace it should feed into didn't.
    csx.Update();

    CSRectGrid* csRectGrid = csx.GetGrid();
    for (int axis = 0; axis < 3; ++axis) {
        unsigned int qty = 0;
        double* lines = csRectGrid->GetLines(axis, nullptr, qty, /*sorted=*/true);
        _rawLines[axis].assign(lines, lines + qty);
        delete[] lines;
    }
    _gridDeltaMetres = csRectGrid->GetDeltaUnit();

    _grid.dims = CopperGridDims{numLines(0), numLines(1), numLines(2)};
    const std::size_t cellCount = _grid.dims.cellCount();
    for (int axis = 0; axis < 3; ++axis) {
        _grid.vv[axis].assign(cellCount, 0.0F);
        _grid.vi[axis].assign(cellCount, 0.0F);
        _grid.ii[axis].assign(cellCount, 0.0F);
        _grid.iv[axis].assign(cellCount, 0.0F);
    }
    _grid.lineX.resize(numLines(0));
    _grid.lineY.resize(numLines(1));
    _grid.lineZ.resize(numLines(2));
    _grid.dualLineX.resize(numLines(0));
    _grid.dualLineY.resize(numLines(1));
    _grid.dualLineZ.resize(numLines(2));
    for (unsigned int i = 0; i < numLines(0); ++i) {
        _grid.lineX[i] = static_cast<float>(discLine(0, i, false) * _gridDeltaMetres);
        _grid.dualLineX[i] = static_cast<float>(discLine(0, i, true) * _gridDeltaMetres);
    }
    for (unsigned int i = 0; i < numLines(1); ++i) {
        _grid.lineY[i] = static_cast<float>(discLine(1, i, false) * _gridDeltaMetres);
        _grid.dualLineY[i] = static_cast<float>(discLine(1, i, true) * _gridDeltaMetres);
    }
    for (unsigned int i = 0; i < numLines(2); ++i) {
        _grid.lineZ[i] = static_cast<float>(discLine(2, i, false) * _gridDeltaMetres);
        _grid.dualLineZ[i] = static_cast<float>(discLine(2, i, true) * _gridDeltaMetres);
    }
    for (int axis = 0; axis < 3; ++axis) {
        _grid.primaryDelta[axis].resize(numLines(axis));
        _grid.dualDelta[axis].resize(numLines(axis));
        for (unsigned int i = 0; i < numLines(axis); ++i) {
            _grid.primaryDelta[axis][i] = discDelta(axis, static_cast<int>(i), false) * _gridDeltaMetres;
            _grid.dualDelta[axis][i] = discDelta(axis, static_cast<int>(i), true) * _gridDeltaMetres;
        }
    }

    computeMaterialCoefficients();
    if (hasConformalGeometry()) {
        applyConformalPEC(calcTimestepVar3());
    }

    double dT = calcTimestepVar3();
    if (hasConformalGeometry()) {
        const auto started = std::chrono::steady_clock::now();
        const double limit = conformalStableTimestep();
        if (std::getenv("COPPER_TRACE_TIMESTEP")) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::fprintf(stderr, "Copper: Var3 timestep %.6g s, conformal stability limit %.6g s (%.4f of Var3), %.2f s\n",
                         dT, limit, limit / dT, seconds);
        }
        dT = std::min(dT, kConformalTimestepSafety * limit);
    }
    _grid.timestepSeconds = dT;
    restoreInertCoefficients();

    for (int n = 0; n < 3; ++n) {
        for (unsigned int z = 0; z < numLines(2); ++z) {
            for (unsigned int y = 0; y < numLines(1); ++y) {
                for (unsigned int x = 0; x < numLines(0); ++x) {
                    const std::size_t i = index(x, y, z);
                    const double c = _ecC[n][i];
                    const double g = _ecG[n][i];
                    if (c > 0.0) {
                        _grid.vv[n][i] = static_cast<float>((1.0 - dT * g / 2.0 / c) / (1.0 + dT * g / 2.0 / c));
                        _grid.vi[n][i] = static_cast<float>((dT / c) / (1.0 + dT * g / 2.0 / c));
                    } else {
                        _grid.vv[n][i] = 0.0F;
                        _grid.vi[n][i] = 0.0F;
                    }
                    const double l = _ecL[n][i];
                    const double r = _ecR[n][i];
                    if (l > 0.0) {
                        _grid.ii[n][i] = static_cast<float>((1.0 - dT * r / 2.0 / l) / (1.0 + dT * r / 2.0 / l));
                        _grid.iv[n][i] = static_cast<float>((dT / l) / (1.0 + dT * r / 2.0 / l));
                    } else {
                        _grid.ii[n][i] = 0.0F;
                        _grid.iv[n][i] = 0.0F;
                    }
                }
            }
        }
    }

    computeBoundaryPEC();
    computePEC();
    zeroConformalPEC();
    zeroOuterHLayer();
    computeParallelLumpedElements();
    computeExcitation();
}

} // namespace copper
