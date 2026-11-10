// The conformal metal treatment (CopperOperator::applyConformalPEC(), CopperConformal.hpp): its cut
// geometry against closed forms, the coefficients it leaves on a cut cylinder, and the property it
// exists for -- a square coax whose round inner conductor sits at different places between the mesh
// lines keeps the same characteristic impedance, as the analytic line does, where a staircased one
// doesn't.
#import <XCTest/XCTest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <tuple>
#include <utility>
#include <vector>

#include <CSPrimBox.h>
#include <CSPrimLinPoly.h>
#include <CSPropExcitation.h>
#include <CSPropMetal.h>
#include <CSRectGrid.h>

#include "CopperFDTDRunner.h"
#include "CopperTestFixtures.hpp"
#include "Internal/CopperCPML.hpp"
#include "Internal/CopperConformal.hpp"
#include "Internal/CopperEngine.hpp"
#include "Internal/CopperOperator.hpp"

// macOS's own headers already define a Polygon (QuickDraw's).
using CutPolygon = copper::conformal::Polygon;
using copper::test::pulseConfig;

namespace {

CutPolygon circle(double cx, double cy, double radius, int segments = 64) {
    CutPolygon polygon;
    for (int i = 0; i < segments; ++i) {
        const double angle = 2 * std::numbers::pi * i / segments;
        polygon.x.push_back(cx + radius * std::cos(angle));
        polygon.y.push_back(cy + radius * std::sin(angle));
    }
    return polygon;
}

// A z-extruded cylinder of `polygon`, from z0 to z1, on its own metal property -- conformal or not.
void addCylinder(ContinuousStructure& csx, const CutPolygon& polygon, double z0, double z1, bool conformal) {
    auto* metal = new CSPropMetal(csx.GetParameterSet());
    metal->SetName("cylinder");
    if (conformal) metal->SetAttributeValue(copper::CopperOperator::kConformalAttribute, "1");
    csx.AddProperty(metal);
    auto* prim = new CSPrimLinPoly(metal->GetParameterSet(), metal);
    prim->ClearCoords();
    for (std::size_t i = 0; i < polygon.x.size(); ++i) {
        prim->AddCoord(polygon.x[i]);
        prim->AddCoord(polygon.y[i]);
    }
    prim->SetNormDir(2);
    prim->SetElevation(z0);
    prim->SetLength(z1 - z0);
    prim->SetPriority(50);
}

// Uniform 1 mm lines from 0 to `extent` along each axis, plus any `extraZ` lines.
ContinuousStructure* uniformGrid(double extentX, double extentY, double extentZ, std::vector<double> extraZ = {}) {
    auto* csx = new ContinuousStructure();
    CSRectGrid* grid = csx->GetGrid();
    grid->SetDeltaUnit(1e-3);
    const double extents[3] = {extentX, extentY, extentZ};
    for (int axis = 0; axis < 3; ++axis) {
        for (double line = 0; line <= extents[axis] + 1e-9; line += 1.0) grid->AddDiscLine(axis, line);
    }
    for (const double line : extraZ) grid->AddDiscLine(2, line);
    return csx;
}

constexpr double kCoaxSide = 30.0;   // mm, the square outer conductor (the domain's PEC walls)
constexpr double kCoaxLength = 160.0;

// Characteristic impedance of a square coax of side `side` with a round centre conductor of diameter
// `diameter`, in vacuum (Wadell, "Transmission Line Design Handbook", 1991): 60 ln(1.0787 W/d).
double squareCoaxImpedance(double side, double diameter) { return 60.0 * std::log(1.0787 * side / diameter); }

// Runs a Gaussian pulse down a square coax whose inner conductor, of `radius` mm, is centred at
// (15 + dx, 15 + dy) mm, and returns the impedance a forward wave carries past a plane 40 mm from the source:
// sqrt(sum V^2 / sum I^2), V the line integral of E from the inner conductor to the wall and I the
// circulation of H around the inner conductor.
double measuredCoaxImpedance(double dx, double dy, bool conformal, double radius) {
    // A board's timestep is set by its finest cells, which a via's neighbourhood rarely holds; one
    // 0.4 mm cell beyond the probe plane gives this line the same headroom.
    std::unique_ptr<ContinuousStructure> csx(uniformGrid(kCoaxSide, kCoaxSide, kCoaxLength, {140.4}));
    const double cx = kCoaxSide / 2 + dx, cy = kCoaxSide / 2 + dy;
    addCylinder(*csx, circle(cx, cy, radius), 0.0, kCoaxLength, conformal);

    // A gap source from the inner conductor's surface to the wall along the row of nodes nearest its
    // centre: it launches the TEM mode, and the higher modes it also excites are cut off below 5 GHz.
    const double row = std::round(cy);
    const double sourceZ = 30.0, probeZ = 70.0;
    auto* exc = new CSPropExcitation(csx->GetParameterSet());
    exc->SetName("gap");
    exc->SetExcitType(0);
    exc->SetExcitation(1.0, 0);
    csx->AddProperty(exc);
    auto* gap = new CSPrimBox(exc->GetParameterSet(), exc);
    const double gapStart[3] = {std::ceil(cx + radius), row, sourceZ};
    const double gapStop[3] = {kCoaxSide, row, sourceZ};
    for (int axis = 0; axis < 3; ++axis) {
        gap->SetCoord(2 * axis, gapStart[axis]);
        gap->SetCoord(2 * axis + 1, gapStop[axis]);
    }

    copper::CopperOperator::Config config;
    config.boundary = {copper::CopperOperator::BoundaryType::PEC,  copper::CopperOperator::BoundaryType::PEC,
                       copper::CopperOperator::BoundaryType::PEC,  copper::CopperOperator::BoundaryType::PEC,
                       copper::CopperOperator::BoundaryType::Open, copper::CopperOperator::BoundaryType::Open};
    config.f0 = 1e9;
    config.fc = 1e9;
    config.maxTimesteps = 4000;
    copper::CopperOperator op(*csx, config);
    const copper::CopperCPML cpml =
        copper::buildCPML(op, copper::cpmlAlphaMaxForFrequency(0.2e9), 8, copper::CopperCPMLFaces::ZOnly);
    copper::CopperEngine engine(op.grid(), op.excitation(), cpml);

    // Mesh indices: lines are at whole millimetres.
    const auto k = static_cast<std::uint32_t>(probeZ);
    const auto j = static_cast<std::uint32_t>(row);
    const auto lo = static_cast<std::uint32_t>(std::floor(cx - radius) - 3);
    const auto hi = static_cast<std::uint32_t>(std::ceil(cx + radius) + 3);
    const auto jLo = static_cast<std::uint32_t>(std::floor(cy - radius) - 3);
    const auto jHi = static_cast<std::uint32_t>(std::ceil(cy + radius) + 3);
    const auto centre = static_cast<std::uint32_t>(std::round(cx));
    double sumV2 = 0.0, sumI2 = 0.0;
    engine.runWithProbeSampling(config.maxTimesteps, [&](std::uint32_t) {
        double v = 0.0;
        for (std::uint32_t i = centre; i < static_cast<std::uint32_t>(kCoaxSide); ++i) {
            v += engine.readFieldCell(copper::CopperEngine::Field::Ex, i, j, k);
        }
        // Counter-clockwise around the rectangle of dual edges halfway between lines lo..hi, jLo..jHi.
        double current = 0.0;
        for (std::uint32_t i = lo + 1; i <= hi; ++i) {
            current += engine.readFieldCell(copper::CopperEngine::Field::Hx, i, jLo, k);
            current -= engine.readFieldCell(copper::CopperEngine::Field::Hx, i, jHi, k);
        }
        for (std::uint32_t jj = jLo + 1; jj <= jHi; ++jj) {
            current += engine.readFieldCell(copper::CopperEngine::Field::Hy, hi, jj, k);
            current -= engine.readFieldCell(copper::CopperEngine::Field::Hy, lo, jj, k);
        }
        sumV2 += v * v;
        sumI2 += current * current;
        return true;
    });
    return std::sqrt(sumV2 / sumI2);
}

} // namespace

@interface CopperConformalTests : XCTestCase
@end

@implementation CopperConformalTests

- (void)testInsideLengthAndAreaOfASquare {
    const CutPolygon square{{0, 10, 10, 0}, {0, 0, 10, 10}};
    XCTAssertEqualWithAccuracy(copper::conformal::insideLength(square, true, 5, -5, 5), 5.0, 1e-12);
    XCTAssertEqualWithAccuracy(copper::conformal::insideLength(square, false, 2, 4, 30), 6.0, 1e-12);
    XCTAssertEqualWithAccuracy(copper::conformal::insideLength(square, true, 12, 0, 10), 0.0, 1e-12);
    XCTAssertEqualWithAccuracy(copper::conformal::insideArea(square, 5, 15, 5, 15), 25.0, 1e-12);
    XCTAssertEqualWithAccuracy(copper::conformal::insideArea(square, -1, 11, -1, 11), 100.0, 1e-12);
    XCTAssertEqualWithAccuracy(copper::conformal::insideArea(square, 20, 30, 0, 10), 0.0, 1e-12);
}

- (void)testCutCellsOfACircleAddUpToItsArea {
    // A circle cut by an off-centre unit mesh: the pieces of every cell sum to the polygon's area,
    // and each chord length matches the circle's to within the 64-gon's sagitta.
    const double radius = 3.3;
    const CutPolygon polygon = circle(0.37, -0.21, radius);
    const double polygonArea = 0.5 * 64 * std::sin(2 * std::numbers::pi / 64) * radius * radius;
    double total = 0.0;
    for (int x = -5; x < 5; ++x) {
        for (int y = -5; y < 5; ++y) total += copper::conformal::insideArea(polygon, x, x + 1, y, y + 1);
    }
    XCTAssertEqualWithAccuracy(total, polygonArea, 1e-9);
    const double chord = copper::conformal::insideLength(polygon, true, 1.0, -10, 10);
    XCTAssertEqualWithAccuracy(chord, 2 * std::sqrt(radius * radius - 1.21 * 1.21), 0.02);
}

- (void)testThinWireFactorIsOneAtTheIntrinsicRadiusOfABareNodeWire {
    // Railton et al. (IEEE TEMC 2006): on a square mesh, an unmodified line of PEC nodes behaves as a
    // wire of radius exp(-pi/2) of the cell, so that is where the correction vanishes.
    XCTAssertEqualWithAccuracy(copper::conformal::thinWirePermeabilityFactor(std::exp(-std::numbers::pi / 2), 1, 1), 1.0,
                               1e-12);
    XCTAssertGreaterThan(copper::conformal::thinWirePermeabilityFactor(0.05, 1, 1), 1.0);
    XCTAssertLessThan(copper::conformal::thinWirePermeabilityFactor(0.4, 1, 1), 1.0);
}

- (void)testConformalCylinderCarriesOnlyItsOpenEdgesAndFaces {
    const auto build = [](bool withCylinder) {
        std::unique_ptr<ContinuousStructure> csx(uniformGrid(10, 10, 4, {3.6}));
        if (withCylinder) addCylinder(*csx, circle(5.0, 5.0, 1.5), 0.0, 4.0, true);
        return csx;
    };
    const auto plain = build(false);
    const auto cut = build(true);
    const copper::CopperOperator bare(*plain, pulseConfig(10));
    const copper::CopperOperator op(*cut, pulseConfig(10));
    const copper::CopperYeeGrid& a = bare.grid();
    const copper::CopperYeeGrid& b = op.grid();
    const auto at = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) { return copperGridIndex(b.dims, x, y, z); };

    // The node at the centre is inside: its Z edge is metal.
    XCTAssertEqual(b.vv[2][at(5, 5, 1)], 0.0F);
    XCTAssertEqual(b.vi[2][at(5, 5, 1)], 0.0F);
    // The X edge from (6,5) to (7,5) is cut at x = 6.5: half of it is open, so its curl coefficient
    // (dT/C, C = eps A / l_open) is half the bare edge's, on the same timestep.
    const double ratio = b.vi[0][at(6, 5, 1)] / a.vi[0][at(6, 5, 1)] * (a.timestepSeconds / b.timestepSeconds);
    XCTAssertEqualWithAccuracy(ratio, 0.5, 1.0 / 128);
    // The X edge from (5,5) to (6,5) is wholly inside: metal.
    XCTAssertEqual(b.vi[0][at(5, 5, 1)], 0.0F);
    // An edge clear of the cylinder is untouched.
    XCTAssertEqualWithAccuracy(b.vi[0][at(1, 1, 1)] / b.timestepSeconds, a.vi[0][at(1, 1, 1)] / a.timestepSeconds,
                               1e-6 * a.vi[0][at(1, 1, 1)] / a.timestepSeconds);
    // The cut faces cost no more timestep than the stability bound allows: with no conformal mode
    // near the limit, only the eigenvalue limit's safety margin.
    XCTAssertGreaterThanOrEqual(b.timestepSeconds, copper::CopperOperator::kConformalTimestepSafety *
                                                       copper::CopperOperator::kConformalStability * a.timestepSeconds);
}

- (void)testBoardViaStaysStableWhereVar3OvershootsItsLimit {
    // A board via (0.3 mm barrel through a 1.51 mm core) on the anisotropic cells a board mesh gives
    // it -- 0.23 mm across, 0.05 mm along, 0.38 mm through. Var3's local estimate allowed ~5% more
    // timestep than the cut faces around this barrel tolerate, and the run diverged within 250
    // steps; CopperOperator now holds the timestep under the leapfrog's eigenvalue limit.
    auto csx = std::make_unique<ContinuousStructure>();
    CSRectGrid* grid = csx->GetGrid();
    grid->SetDeltaUnit(1e-7); // libkiems' simulation unit, 0.1 um
    for (int i = 0; i < 11; ++i) grid->AddDiscLine(0, 94353 + i * 2318.6);
    for (int i = 0; i < 52; ++i) grid->AddDiscLine(1, 83567 + i * 501.84);
    for (int i = 0; i < 5; ++i) grid->AddDiscLine(2, -15100 + i * 3775.0);
    addCylinder(*csx, circle(105146.4, 96500, 3000), -15100, 0, true);

    auto* exc = new CSPropExcitation(csx->GetParameterSet());
    exc->SetName("source");
    exc->SetExcitType(0);
    exc->SetExcitation(1.0, 2);
    csx->AddProperty(exc);
    auto* source = new CSPrimBox(exc->GetParameterSet(), exc);
    const double start[3] = {94353 + 2 * 2318.6, 83567 + 2 * 501.84, -15100};
    const double stop[3] = {start[0], start[1], 0};
    for (int axis = 0; axis < 3; ++axis) {
        source->SetCoord(2 * axis, start[axis]);
        source->SetCoord(2 * axis + 1, stop[axis]);
    }

    copper::CopperOperator::Config config = pulseConfig(10000);
    config.f0 = 3e9;
    config.fc = 3e9;
    const copper::CopperOperator op(*csx, config);
    copper::CopperEngine engine(op.grid(), op.excitation(), {}, copper::CopperEngine::Backend::CPU);
    const auto largestE = [&] {
        double largest = 0.0;
        for (const auto field : {copper::CopperEngine::Field::Ex, copper::CopperEngine::Field::Ey,
                                 copper::CopperEngine::Field::Ez}) {
            for (const float value : engine.readField(field)) {
                if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
                largest = std::max(largest, static_cast<double>(std::fabs(value)));
            }
        }
        return largest;
    };
    engine.run(7000); // the 3 GHz Gaussian is over by ~6200 steps
    const double afterPulse = largestE();
    XCTAssertTrue(std::isfinite(afterPulse));
    XCTAssertGreaterThan(afterPulse, 0.0);
    engine.run(3000);
    // A lossless PEC cavity rings on; it must not grow.
    XCTAssertLessThan(largestE(), 10 * afterPulse);
}

- (void)testSquareCoaxImpedanceDoesNotDependOnWhereTheConductorFallsBetweenLines {
    // Radii of 4 cells and of 1.5 -- the latter like a PCB via's barrel against the mesh a board's
    // traces need -- each at four places between the mesh lines. Conformal or staircased, the error
    // left at 1.5 cells is the Dey-Mittra scheme's own at that resolution (Railton et al. 2006 show
    // the same position-dependent ripple); what conformal removes is the staircase's much larger one.
    const std::pair<double, double> offsets[] = {{0.0, 0.0}, {0.31, 0.17}, {0.5, 0.5}, {0.83, 0.44}};
    for (const auto& [radius, worstError, worstSpread] :
         {std::tuple{4.0, 0.02, 0.015}, std::tuple{1.5, 0.02, 0.025}}) {
        const double analytic = squareCoaxImpedance(kCoaxSide, 2 * radius);
        double conformalLo = 1e9, conformalHi = 0, staircaseLo = 1e9, staircaseHi = 0;
        for (const auto& [dx, dy] : offsets) {
            const double conformal = measuredCoaxImpedance(dx, dy, true, radius);
            const double staircase = measuredCoaxImpedance(dx, dy, false, radius);
            NSLog(@"square coax, radius %.1f mm, offset (%.2f, %.2f) mm: conformal %.2f ohm, staircase %.2f ohm, analytic "
                  @"%.2f ohm",
                  radius, dx, dy, conformal, staircase, analytic);
            conformalLo = std::min(conformalLo, conformal);
            conformalHi = std::max(conformalHi, conformal);
            staircaseLo = std::min(staircaseLo, staircase);
            staircaseHi = std::max(staircaseHi, staircase);
        }
        const double conformalError = std::max(std::abs(conformalLo - analytic), std::abs(conformalHi - analytic));
        const double staircaseError = std::max(std::abs(staircaseLo - analytic), std::abs(staircaseHi - analytic));
        XCTAssertLessThan(conformalError / analytic, worstError, @"radius %.1f", radius);
        XCTAssertLessThan((conformalHi - conformalLo) / analytic, worstSpread, @"radius %.1f: the line moves with the mesh",
                          radius);
        XCTAssertLessThan(conformalError, staircaseError, @"radius %.1f", radius);
        XCTAssertLessThan(conformalHi - conformalLo, staircaseHi - staircaseLo, @"radius %.1f", radius);
    }
}

@end
