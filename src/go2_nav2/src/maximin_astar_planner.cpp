#include "go2_nav2/maximin_astar_planner.hpp"

#include <cmath>
#include <mutex>
#include <string>

#include "go2_nav2/maximin_astar.hpp"
#include "nav2_core/planner_exceptions.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace go2_nav2
{
namespace
{

GridSpec gridFromCostmap(
  const nav2_costmap_2d::Costmap2D & costmap,
  bool allow_unknown)
{
  GridSpec grid;
  grid.width = costmap.getSizeInCellsX();
  grid.height = costmap.getSizeInCellsY();
  grid.resolution = costmap.getResolution();
  grid.origin_x = costmap.getOriginX();
  grid.origin_y = costmap.getOriginY();
  grid.blocked.assign(static_cast<std::size_t>(grid.width) * grid.height, 0);

  for (unsigned int y = 0; y < grid.height; ++y) {
    for (unsigned int x = 0; x < grid.width; ++x) {
      const unsigned char cost = costmap.getCost(x, y);
      bool blocked = false;
      if (cost == nav2_costmap_2d::NO_INFORMATION) {
        blocked = !allow_unknown;
      } else {
        blocked = cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
      }
      grid.blocked[grid.index(x, y)] = blocked ? 1 : 0;
    }
  }

  return grid;
}

double yawBetween(
  const GridSpec & grid,
  const GridCell & from,
  const GridCell & to)
{
  const auto from_world = mapToWorld(grid, from);
  const auto to_world = mapToWorld(grid, to);
  return std::atan2(to_world.second - from_world.second, to_world.first - from_world.first);
}

geometry_msgs::msg::Quaternion orientationForPathPoint(
  const GridSpec & grid,
  const std::vector<GridCell> & path,
  std::size_t index,
  const geometry_msgs::msg::PoseStamped & goal,
  bool use_final_approach_orientation)
{
  if (path.empty()) {
    return goal.pose.orientation;
  }
  if (index + 1 < path.size()) {
    return nav2_util::geometry_utils::orientationAroundZAxis(
      yawBetween(grid, path[index], path[index + 1]));
  }
  if (use_final_approach_orientation) {
    return goal.pose.orientation;
  }
  if (index > 0) {
    return nav2_util::geometry_utils::orientationAroundZAxis(
      yawBetween(grid, path[index - 1], path[index]));
  }
  return goal.pose.orientation;
}

}  // namespace

void MaximinAStarPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  if (!node_) {
    throw nav2_core::PlannerException("Unable to lock lifecycle node.");
  }

  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  global_frame_ = costmap_ros_->getGlobalFrameID();

  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".tolerance", rclcpp::ParameterValue(0.0));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".allow_unknown", rclcpp::ParameterValue(false));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".use_final_approach_orientation", rclcpp::ParameterValue(false));

  node_->get_parameter(name_ + ".tolerance", tolerance_);
  node_->get_parameter(name_ + ".allow_unknown", allow_unknown_);
  node_->get_parameter(name_ + ".use_final_approach_orientation", use_final_approach_orientation_);

  RCLCPP_INFO(
    node_->get_logger(),
    "Configured %s with tolerance=%.3f, allow_unknown=%s",
    name_.c_str(),
    tolerance_,
    allow_unknown_ ? "true" : "false");
}

void MaximinAStarPlanner::cleanup()
{
  RCLCPP_INFO(node_->get_logger(), "Cleaning up %s", name_.c_str());
}

void MaximinAStarPlanner::activate()
{
  RCLCPP_INFO(node_->get_logger(), "Activating %s", name_.c_str());
}

void MaximinAStarPlanner::deactivate()
{
  RCLCPP_INFO(node_->get_logger(), "Deactivating %s", name_.c_str());
}

nav_msgs::msg::Path MaximinAStarPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal,
  std::function<bool()> cancel_checker)
{
  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  if (costmap == nullptr) {
    throw nav2_core::PlannerException("Costmap is not available.");
  }

  GridSpec grid;
  {
    std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getMutex()));
    grid = gridFromCostmap(*costmap, allow_unknown_);
  }

  unsigned int start_x = 0;
  unsigned int start_y = 0;
  unsigned int goal_x = 0;
  unsigned int goal_y = 0;
  if (!costmap->worldToMap(start.pose.position.x, start.pose.position.y, start_x, start_y)) {
    throw nav2_core::StartOutsideMapBounds("Start is outside the global costmap.");
  }
  if (!costmap->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_x, goal_y)) {
    throw nav2_core::GoalOutsideMapBounds("Goal is outside the global costmap.");
  }

  const GridCell start_cell{start_x, start_y};
  const GridCell goal_cell{goal_x, goal_y};
  if (grid.isBlocked(start_cell.x, start_cell.y)) {
    throw nav2_core::StartOccupied("Start is inside occupied space.");
  }

  const MaximinPlanResult result =
    planMaximinClearancePath(grid, start_cell, goal_cell, tolerance_, cancel_checker);

  if (!result.success) {
    if (result.message == "Planning was canceled.") {
      throw nav2_core::PlannerCancelled(result.message);
    }
    if (result.message == "Goal is inside occupied space.") {
      throw nav2_core::GoalOccupied(result.message);
    }
    throw nav2_core::NoValidPathCouldBeFound(result.message);
  }

  nav_msgs::msg::Path path;
  path.header.stamp = node_->now();
  path.header.frame_id = global_frame_;
  path.poses.reserve(result.cells.size());

  for (std::size_t index = 0; index < result.cells.size(); ++index) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    const auto world = mapToWorld(grid, result.cells[index]);
    pose.pose.position.x = world.first;
    pose.pose.position.y = world.second;
    pose.pose.position.z = 0.0;

    if (index + 1 == result.cells.size() && result.target == goal_cell) {
      pose.pose.position.x = goal.pose.position.x;
      pose.pose.position.y = goal.pose.position.y;
    }

    pose.pose.orientation = orientationForPathPoint(
      grid, result.cells, index, goal, use_final_approach_orientation_);
    path.poses.push_back(pose);
  }

  if (path.poses.empty()) {
    throw nav2_core::NoValidPathCouldBeFound("Planner returned an empty path.");
  }

  return path;
}

}  // namespace go2_nav2

PLUGINLIB_EXPORT_CLASS(go2_nav2::MaximinAStarPlanner, nav2_core::GlobalPlanner)
