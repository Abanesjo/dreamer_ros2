#ifndef GO2_NAV2__MAXIMIN_ASTAR_HPP_
#define GO2_NAV2__MAXIMIN_ASTAR_HPP_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace go2_nav2
{

struct GridCell
{
  unsigned int x{0};
  unsigned int y{0};

  bool operator==(const GridCell & other) const
  {
    return x == other.x && y == other.y;
  }
};

struct GridSpec
{
  unsigned int width{0};
  unsigned int height{0};
  double resolution{1.0};
  double origin_x{0.0};
  double origin_y{0.0};
  std::vector<uint8_t> blocked;

  bool valid() const;
  bool inBounds(int x, int y) const;
  std::size_t index(unsigned int x, unsigned int y) const;
  bool isBlocked(unsigned int x, unsigned int y) const;
  bool isFree(unsigned int x, unsigned int y) const;
};

struct MaximinPlanResult
{
  bool success{false};
  std::string message;
  std::vector<GridCell> cells;
  GridCell target;
};

std::optional<GridCell> worldToMap(const GridSpec & grid, double wx, double wy);

std::pair<double, double> mapToWorld(const GridSpec & grid, const GridCell & cell);

std::vector<uint8_t> inflateBlockedCells(
  const std::vector<uint8_t> & blocked,
  unsigned int width,
  unsigned int height,
  double inflation_radius_m,
  double resolution);

std::vector<double> computeClearanceMap(const GridSpec & grid);

MaximinPlanResult planMaximinClearancePath(
  const GridSpec & grid,
  const GridCell & start,
  const GridCell & goal,
  double goal_tolerance_m = 0.0,
  const std::function<bool()> & cancel_checker = {},
  double clearance_target_m = 0.0);

}  // namespace go2_nav2

#endif  // GO2_NAV2__MAXIMIN_ASTAR_HPP_
