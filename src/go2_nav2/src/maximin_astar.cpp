#include "go2_nav2/maximin_astar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace go2_nav2
{
namespace
{

constexpr double kEpsilon = 1.0e-9;
constexpr std::size_t kNoParent = std::numeric_limits<std::size_t>::max();

struct Neighbor
{
  int dx;
  int dy;
  double cost;
};

const std::vector<Neighbor> & neighbors8()
{
  static const std::vector<Neighbor> neighbors = {
    {1, 0, 1.0},
    {-1, 0, 1.0},
    {0, 1, 1.0},
    {0, -1, 1.0},
    {1, 1, std::sqrt(2.0)},
    {1, -1, std::sqrt(2.0)},
    {-1, 1, std::sqrt(2.0)},
    {-1, -1, std::sqrt(2.0)},
  };
  return neighbors;
}

bool sameClearance(double lhs, double rhs)
{
  if (std::isinf(lhs) && std::isinf(rhs)) {
    return true;
  }
  return std::abs(lhs - rhs) <= kEpsilon;
}

double scoreClearance(double clearance, double clearance_target_m)
{
  if (clearance_target_m <= 0.0) {
    return clearance;
  }
  return std::min(clearance, clearance_target_m);
}

std::vector<double> scoreClearances(
  const std::vector<double> & clearances,
  double clearance_target_m)
{
  if (clearance_target_m <= 0.0) {
    return clearances;
  }

  std::vector<double> scored;
  scored.reserve(clearances.size());
  for (const double clearance : clearances) {
    scored.push_back(scoreClearance(clearance, clearance_target_m));
  }
  return scored;
}

bool betterLabel(
  double candidate_clearance,
  double candidate_length,
  double best_clearance,
  double best_length)
{
  if (candidate_clearance > best_clearance + kEpsilon) {
    return true;
  }
  if (sameClearance(candidate_clearance, best_clearance) &&
    candidate_length + kEpsilon < best_length)
  {
    return true;
  }
  return false;
}

bool diagonalMoveIsClear(const GridSpec & grid, const GridCell & current, const Neighbor & step)
{
  if (step.dx == 0 || step.dy == 0) {
    return true;
  }

  const int adjacent_x_a = static_cast<int>(current.x) + step.dx;
  const int adjacent_y_a = static_cast<int>(current.y);
  const int adjacent_x_b = static_cast<int>(current.x);
  const int adjacent_y_b = static_cast<int>(current.y) + step.dy;

  return grid.inBounds(adjacent_x_a, adjacent_y_a) &&
         grid.inBounds(adjacent_x_b, adjacent_y_b) &&
         grid.isFree(static_cast<unsigned int>(adjacent_x_a),
        static_cast<unsigned int>(adjacent_y_a)) &&
         grid.isFree(static_cast<unsigned int>(adjacent_x_b),
        static_cast<unsigned int>(adjacent_y_b));
}

struct ClearanceQueueEntry
{
  double distance;
  std::size_t index;
};

struct ClearanceQueueCompare
{
  bool operator()(const ClearanceQueueEntry & lhs, const ClearanceQueueEntry & rhs) const
  {
    return lhs.distance > rhs.distance;
  }
};

struct GoalCandidate
{
  GridCell cell;
  double distance_to_goal;
  double clearance;
};

std::vector<GoalCandidate> findGoalCandidates(
  const GridSpec & grid,
  const std::vector<double> & clearances,
  const GridCell & goal,
  double goal_tolerance_m)
{
  std::vector<GoalCandidate> candidates;
  if (!grid.inBounds(static_cast<int>(goal.x), static_cast<int>(goal.y))) {
    return candidates;
  }

  const int radius_cells = goal_tolerance_m > 0.0 ?
    static_cast<int>(std::ceil(goal_tolerance_m / grid.resolution)) : 0;

  const auto goal_world = mapToWorld(grid, goal);
  for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
      const int x = static_cast<int>(goal.x) + dx;
      const int y = static_cast<int>(goal.y) + dy;
      if (!grid.inBounds(x, y)) {
        continue;
      }
      const GridCell candidate{static_cast<unsigned int>(x), static_cast<unsigned int>(y)};
      if (!grid.isFree(candidate.x, candidate.y)) {
        continue;
      }
      const auto candidate_world = mapToWorld(grid, candidate);
      const double distance_to_goal = std::hypot(
        candidate_world.first - goal_world.first,
        candidate_world.second - goal_world.second);
      if (distance_to_goal > goal_tolerance_m + kEpsilon) {
        continue;
      }
      candidates.push_back(
        GoalCandidate{
            candidate,
            distance_to_goal,
            clearances[grid.index(candidate.x, candidate.y)]});
    }
  }

  std::sort(
    candidates.begin(), candidates.end(),
    [](const GoalCandidate & lhs, const GoalCandidate & rhs) {
      if (std::abs(lhs.distance_to_goal - rhs.distance_to_goal) > kEpsilon) {
        return lhs.distance_to_goal < rhs.distance_to_goal;
      }
      if (!sameClearance(lhs.clearance, rhs.clearance)) {
        return lhs.clearance > rhs.clearance;
      }
      if (lhs.cell.y != rhs.cell.y) {
        return lhs.cell.y < rhs.cell.y;
      }
      return lhs.cell.x < rhs.cell.x;
    });

  return candidates;
}

struct SearchQueueEntry
{
  double clearance;
  double length;
  std::size_t index;
};

struct SearchQueueCompare
{
  bool operator()(const SearchQueueEntry & lhs, const SearchQueueEntry & rhs) const
  {
    if (!sameClearance(lhs.clearance, rhs.clearance)) {
      return lhs.clearance < rhs.clearance;
    }
    return lhs.length > rhs.length;
  }
};

std::vector<GridCell> reconstructPath(
  const GridSpec & grid,
  const std::vector<std::size_t> & parent,
  std::size_t current)
{
  std::vector<GridCell> path;
  while (current != kNoParent) {
    const unsigned int x = static_cast<unsigned int>(current % grid.width);
    const unsigned int y = static_cast<unsigned int>(current / grid.width);
    path.push_back(GridCell{x, y});
    current = parent[current];
  }
  std::reverse(path.begin(), path.end());
  return path;
}

MaximinPlanResult planToFixedGoal(
  const GridSpec & grid,
  const std::vector<double> & clearances,
  const GridCell & start,
  const GridCell & goal,
  const std::function<bool()> & cancel_checker)
{
  const std::size_t size = static_cast<std::size_t>(grid.width) * grid.height;
  const std::size_t start_index = grid.index(start.x, start.y);
  const std::size_t goal_index = grid.index(goal.x, goal.y);

  std::vector<double> best_clearance(size, -1.0);
  std::vector<double> best_length(size, std::numeric_limits<double>::infinity());
  std::vector<std::size_t> parent(size, kNoParent);
  std::priority_queue<SearchQueueEntry, std::vector<SearchQueueEntry>, SearchQueueCompare> open;

  best_clearance[start_index] = clearances[start_index];
  best_length[start_index] = 0.0;
  open.push(SearchQueueEntry{best_clearance[start_index], 0.0, start_index});

  while (!open.empty()) {
    if (cancel_checker && cancel_checker()) {
      return MaximinPlanResult{false, "Planning was canceled.", {}, goal};
    }

    const SearchQueueEntry current = open.top();
    open.pop();

    if (!sameClearance(current.clearance, best_clearance[current.index]) ||
      current.length > best_length[current.index] + kEpsilon)
    {
      continue;
    }

    if (current.index == goal_index) {
      return MaximinPlanResult{
        true,
        "Planned maximin-clearance path.",
        reconstructPath(grid, parent, current.index),
        goal};
    }

    const GridCell current_cell{
      static_cast<unsigned int>(current.index % grid.width),
      static_cast<unsigned int>(current.index / grid.width)};

    for (const Neighbor & step : neighbors8()) {
      const int nx = static_cast<int>(current_cell.x) + step.dx;
      const int ny = static_cast<int>(current_cell.y) + step.dy;
      if (!grid.inBounds(nx, ny)) {
        continue;
      }

      const GridCell next_cell{static_cast<unsigned int>(nx), static_cast<unsigned int>(ny)};
      if (!grid.isFree(next_cell.x, next_cell.y) || !diagonalMoveIsClear(grid, current_cell,
            step))
      {
        continue;
      }

      const std::size_t next_index = grid.index(next_cell.x, next_cell.y);
      const double next_clearance = std::min(current.clearance, clearances[next_index]);
      const double next_length = current.length + step.cost * grid.resolution;

      if (!betterLabel(
          next_clearance, next_length, best_clearance[next_index], best_length[next_index]))
      {
        continue;
      }

      best_clearance[next_index] = next_clearance;
      best_length[next_index] = next_length;
      parent[next_index] = current.index;
      open.push(SearchQueueEntry{next_clearance, next_length, next_index});
    }
  }

  return MaximinPlanResult{false, "No path could be found.", {}, goal};
}

}  // namespace

bool GridSpec::valid() const
{
  return width > 0 && height > 0 && resolution > 0.0 &&
         blocked.size() == static_cast<std::size_t>(width) * height;
}

bool GridSpec::inBounds(int x, int y) const
{
  return x >= 0 && y >= 0 && x < static_cast<int>(width) && y < static_cast<int>(height);
}

std::size_t GridSpec::index(unsigned int x, unsigned int y) const
{
  return static_cast<std::size_t>(y) * width + x;
}

bool GridSpec::isBlocked(unsigned int x, unsigned int y) const
{
  return blocked[index(x, y)] != 0;
}

bool GridSpec::isFree(unsigned int x, unsigned int y) const
{
  return !isBlocked(x, y);
}

std::optional<GridCell> worldToMap(const GridSpec & grid, double wx, double wy)
{
  if (!grid.valid() || wx < grid.origin_x || wy < grid.origin_y) {
    return std::nullopt;
  }

  const int mx = static_cast<int>(std::floor((wx - grid.origin_x) / grid.resolution));
  const int my = static_cast<int>(std::floor((wy - grid.origin_y) / grid.resolution));
  if (!grid.inBounds(mx, my)) {
    return std::nullopt;
  }
  return GridCell{static_cast<unsigned int>(mx), static_cast<unsigned int>(my)};
}

std::pair<double, double> mapToWorld(const GridSpec & grid, const GridCell & cell)
{
  return {
    grid.origin_x + (static_cast<double>(cell.x) + 0.5) * grid.resolution,
    grid.origin_y + (static_cast<double>(cell.y) + 0.5) * grid.resolution};
}

std::vector<uint8_t> inflateBlockedCells(
  const std::vector<uint8_t> & blocked,
  unsigned int width,
  unsigned int height,
  double inflation_radius_m,
  double resolution)
{
  if (inflation_radius_m <= 0.0 || resolution <= 0.0) {
    return blocked;
  }

  std::vector<uint8_t> inflated = blocked;
  const int radius_cells = static_cast<int>(std::ceil(inflation_radius_m / resolution));
  std::vector<std::pair<int, int>> offsets;
  for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
      if (std::hypot(static_cast<double>(dx), static_cast<double>(dy)) * resolution <=
        inflation_radius_m + kEpsilon)
      {
        offsets.emplace_back(dx, dy);
      }
    }
  }

  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      const std::size_t source_index = static_cast<std::size_t>(y) * width + x;
      if (blocked[source_index] == 0) {
        continue;
      }
      for (const auto & offset : offsets) {
        const int nx = static_cast<int>(x) + offset.first;
        const int ny = static_cast<int>(y) + offset.second;
        if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) {
          continue;
        }
        inflated[static_cast<std::size_t>(ny) * width + static_cast<unsigned int>(nx)] = 1;
      }
    }
  }

  return inflated;
}

std::vector<double> computeClearanceMap(const GridSpec & grid)
{
  const std::size_t size = static_cast<std::size_t>(grid.width) * grid.height;
  std::vector<double> clearances(size, std::numeric_limits<double>::infinity());
  std::priority_queue<
    ClearanceQueueEntry,
    std::vector<ClearanceQueueEntry>,
    ClearanceQueueCompare> open;

  for (std::size_t index = 0; index < size; ++index) {
    if (grid.blocked[index] != 0) {
      clearances[index] = 0.0;
      open.push(ClearanceQueueEntry{0.0, index});
    }
  }

  while (!open.empty()) {
    const ClearanceQueueEntry current = open.top();
    open.pop();
    if (current.distance > clearances[current.index] + kEpsilon) {
      continue;
    }

    const GridCell current_cell{
      static_cast<unsigned int>(current.index % grid.width),
      static_cast<unsigned int>(current.index / grid.width)};

    for (const Neighbor & step : neighbors8()) {
      const int nx = static_cast<int>(current_cell.x) + step.dx;
      const int ny = static_cast<int>(current_cell.y) + step.dy;
      if (!grid.inBounds(nx, ny)) {
        continue;
      }

      const std::size_t next_index =
        static_cast<std::size_t>(static_cast<unsigned int>(ny)) * grid.width +
        static_cast<unsigned int>(nx);
      const double next_distance = current.distance + step.cost * grid.resolution;
      if (next_distance + kEpsilon >= clearances[next_index]) {
        continue;
      }
      clearances[next_index] = next_distance;
      open.push(ClearanceQueueEntry{next_distance, next_index});
    }
  }

  return clearances;
}

MaximinPlanResult planMaximinClearancePath(
  const GridSpec & grid,
  const GridCell & start,
  const GridCell & goal,
  double goal_tolerance_m,
  const std::function<bool()> & cancel_checker,
  double clearance_target_m)
{
  if (!grid.valid()) {
    return MaximinPlanResult{false, "Planning grid is invalid.", {}, goal};
  }
  if (!grid.inBounds(static_cast<int>(start.x), static_cast<int>(start.y))) {
    return MaximinPlanResult{false, "Start is outside the planning grid.", {}, goal};
  }
  if (!grid.inBounds(static_cast<int>(goal.x), static_cast<int>(goal.y))) {
    return MaximinPlanResult{false, "Goal is outside the planning grid.", {}, goal};
  }
  if (grid.isBlocked(start.x, start.y)) {
    return MaximinPlanResult{false, "Start is inside occupied space.", {}, goal};
  }

  const std::vector<double> clearances = computeClearanceMap(grid);
  const std::vector<double> scored_clearances = scoreClearances(clearances, clearance_target_m);
  const std::vector<GoalCandidate> candidates =
    findGoalCandidates(grid, scored_clearances, goal, std::max(0.0, goal_tolerance_m));

  if (candidates.empty()) {
    return MaximinPlanResult{false, "Goal is inside occupied space.", {}, goal};
  }

  for (const GoalCandidate & candidate : candidates) {
    MaximinPlanResult result =
      planToFixedGoal(grid, scored_clearances, start, candidate.cell, cancel_checker);
    if (result.success) {
      return result;
    }
    if (result.message == "Planning was canceled.") {
      return result;
    }
  }

  return MaximinPlanResult{false, "No path could be found.", {}, goal};
}

}  // namespace go2_nav2
