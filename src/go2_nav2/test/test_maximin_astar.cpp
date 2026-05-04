#include <algorithm>
#include <vector>

#include "go2_nav2/maximin_astar.hpp"
#include "gtest/gtest.h"

namespace
{

go2_nav2::GridSpec makeGrid(unsigned int width, unsigned int height)
{
  go2_nav2::GridSpec grid;
  grid.width = width;
  grid.height = height;
  grid.resolution = 1.0;
  grid.origin_x = 0.0;
  grid.origin_y = 0.0;
  grid.blocked.assign(static_cast<std::size_t>(width) * height, 0);
  return grid;
}

void block(go2_nav2::GridSpec & grid, unsigned int x, unsigned int y)
{
  grid.blocked[grid.index(x, y)] = 1;
}

void blockBoundaries(go2_nav2::GridSpec & grid)
{
  for (unsigned int x = 0; x < grid.width; ++x) {
    block(grid, x, 0);
    block(grid, x, grid.height - 1);
  }
  for (unsigned int y = 0; y < grid.height; ++y) {
    block(grid, 0, y);
    block(grid, grid.width - 1, y);
  }
}

}  // namespace

TEST(MaximinAStar, KeepsCenterlineInStraightHallway)
{
  auto grid = makeGrid(9, 7);
  for (unsigned int x = 0; x < grid.width; ++x) {
    block(grid, x, 0);
    block(grid, x, 6);
  }

  const auto result = go2_nav2::planMaximinClearancePath(
    grid,
    go2_nav2::GridCell{1, 3},
    go2_nav2::GridCell{7, 3});

  ASSERT_TRUE(result.success) << result.message;
  ASSERT_FALSE(result.cells.empty());
  EXPECT_TRUE(std::all_of(
    result.cells.begin(), result.cells.end(),
      [](const go2_nav2::GridCell & cell) {return cell.y == 3;}));
}

TEST(MaximinAStar, ChoosesWiderGateOverShorterNarrowGate)
{
  auto grid = makeGrid(15, 9);
  blockBoundaries(grid);

  for (unsigned int y = 1; y < grid.height - 1; ++y) {
    if (y == 2 || y == 5 || y == 6 || y == 7) {
      continue;
    }
    block(grid, 7, y);
  }

  const auto result = go2_nav2::planMaximinClearancePath(
    grid,
    go2_nav2::GridCell{2, 2},
    go2_nav2::GridCell{12, 2});

  ASSERT_TRUE(result.success) << result.message;

  bool crossed_wide_gate = false;
  bool crossed_narrow_gate = false;
  for (const auto & cell : result.cells) {
    if (cell.x != 7) {
      continue;
    }
    crossed_wide_gate = crossed_wide_gate || cell.y == 6;
    crossed_narrow_gate = crossed_narrow_gate || cell.y == 2;
  }

  EXPECT_TRUE(crossed_wide_gate);
  EXPECT_FALSE(crossed_narrow_gate);
}

TEST(MaximinAStar, DoesNotCutDiagonalCorners)
{
  auto grid = makeGrid(2, 2);
  block(grid, 1, 0);
  block(grid, 0, 1);

  const auto result = go2_nav2::planMaximinClearancePath(
    grid,
    go2_nav2::GridCell{0, 0},
    go2_nav2::GridCell{1, 1});

  EXPECT_FALSE(result.success);
}

TEST(MaximinAStar, RejectsBlockedStart)
{
  auto grid = makeGrid(3, 3);
  block(grid, 1, 1);

  const auto result = go2_nav2::planMaximinClearancePath(
    grid,
    go2_nav2::GridCell{1, 1},
    go2_nav2::GridCell{2, 2});

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.message, "Start is inside occupied space.");
}
