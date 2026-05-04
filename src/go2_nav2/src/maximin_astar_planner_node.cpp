#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <optional>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "go2_nav2/maximin_astar.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

namespace go2_nav2
{
namespace
{

double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
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
  std::size_t index)
{
  if (index + 1 < path.size()) {
    return nav2_util::geometry_utils::orientationAroundZAxis(
      yawBetween(grid, path[index], path[index + 1]));
  }
  if (index > 0) {
    return nav2_util::geometry_utils::orientationAroundZAxis(
      yawBetween(grid, path[index - 1], path[index]));
  }
  return nav2_util::geometry_utils::orientationAroundZAxis(0.0);
}

}  // namespace

class MaximinAStarPlannerNode : public rclcpp::Node
{
public:
  MaximinAStarPlannerNode()
  : Node("maximin_astar_planner_node")
  {
    declareParameters();
    loadParameters();

    rclcpp::QoS transient_qos(1);
    transient_qos.transient_local().reliable();

    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic_, transient_qos,
      std::bind(&MaximinAStarPlannerNode::onMap, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::SystemDefaultsQoS(),
      std::bind(&MaximinAStarPlannerNode::onOdom, this, std::placeholders::_1));
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      goal_topic_, rclcpp::SystemDefaultsQoS(),
      std::bind(&MaximinAStarPlannerNode::onGoal, this, std::placeholders::_1));
    path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic_, transient_qos);

    if (path_republish_frequency_ > 0.0) {
      republish_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / path_republish_frequency_),
        std::bind(&MaximinAStarPlannerNode::onRepublishTimer, this));
    }

    RCLCPP_INFO(
      get_logger(),
      "Maximin A* planner ready: map=%s, goal=%s, odom=%s, path=%s",
      map_topic_.c_str(),
      goal_topic_.c_str(),
      odom_topic_.c_str(),
      path_topic_.c_str());
  }

private:
  void declareParameters()
  {
    declare_parameter("frame_id", "odom");
    declare_parameter("map_topic", "/map");
    declare_parameter("goal_topic", "/goal_pose");
    declare_parameter("odom_topic", "/odom");
    declare_parameter("path_topic", "/path");
    declare_parameter("path_republish_frequency", 1.0);
    declare_parameter("robot_radius", 0.4);
    declare_parameter("inflation_margin", 0.05);
    declare_parameter("map_occupied_threshold", 50);
    declare_parameter("treat_unknown_as_occupied", true);
    declare_parameter("planner_tolerance", 0.0);
    declare_parameter("clearance_target", 0.60);
  }

  void loadParameters()
  {
    frame_id_ = get_parameter("frame_id").as_string();
    map_topic_ = get_parameter("map_topic").as_string();
    goal_topic_ = get_parameter("goal_topic").as_string();
    odom_topic_ = get_parameter("odom_topic").as_string();
    path_topic_ = get_parameter("path_topic").as_string();
    path_republish_frequency_ = get_parameter("path_republish_frequency").as_double();
    robot_radius_ = get_parameter("robot_radius").as_double();
    inflation_margin_ = get_parameter("inflation_margin").as_double();
    map_occupied_threshold_ = get_parameter("map_occupied_threshold").as_int();
    treat_unknown_as_occupied_ = get_parameter("treat_unknown_as_occupied").as_bool();
    planner_tolerance_ = get_parameter("planner_tolerance").as_double();
    clearance_target_ = get_parameter("clearance_target").as_double();
  }

  void onMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    const double origin_yaw = yawFromQuaternion(msg->info.origin.orientation);
    if (std::abs(origin_yaw) > 1.0e-6 && !reported_map_yaw_) {
      RCLCPP_WARN(
        get_logger(),
        "Map origin yaw is non-zero; this node assumes axis-aligned OccupancyGrid data.");
      reported_map_yaw_ = true;
    }

    GridSpec grid;
    grid.width = msg->info.width;
    grid.height = msg->info.height;
    grid.resolution = msg->info.resolution;
    grid.origin_x = msg->info.origin.position.x;
    grid.origin_y = msg->info.origin.position.y;
    grid.blocked.assign(static_cast<std::size_t>(grid.width) * grid.height, 0);

    for (std::size_t index = 0; index < msg->data.size() && index < grid.blocked.size(); ++index) {
      const int value = static_cast<int>(msg->data[index]);
      const bool unknown = value < 0;
      grid.blocked[index] =
        (unknown && treat_unknown_as_occupied_) || value >= map_occupied_threshold_ ? 1 : 0;
    }

    grid.blocked = inflateBlockedCells(
      grid.blocked,
      grid.width,
      grid.height,
      robot_radius_ + inflation_margin_,
      grid.resolution);
    grid_ = grid;
    tryPlanPendingGoal();
  }

  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    robot_xy_ = {
      msg->pose.pose.position.x,
      msg->pose.pose.position.y};
    tryPlanPendingGoal();
  }

  void onGoal(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    pending_goal_xy_ = {
      msg->pose.position.x,
      msg->pose.position.y};

    if (!grid_.has_value() || !robot_xy_.has_value()) {
      has_path_ = false;
      publishEmptyPath();
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Goal received; waiting for map and odom before one-shot maximin A* planning.");
      return;
    }

    tryPlanPendingGoal();
  }

  void onRepublishTimer()
  {
    tryPlanPendingGoal();
    if (!has_path_) {
      return;
    }
    last_path_.header.stamp = now();
    for (auto & pose : last_path_.poses) {
      pose.header = last_path_.header;
    }
    path_pub_->publish(last_path_);
  }

  void tryPlanPendingGoal()
  {
    if (!pending_goal_xy_.has_value() || !grid_.has_value() || !robot_xy_.has_value()) {
      return;
    }

    const GridSpec & grid = grid_.value();
    const auto start_cell = worldToMap(grid, robot_xy_->at(0), robot_xy_->at(1));
    const auto goal_cell = worldToMap(grid, pending_goal_xy_->at(0), pending_goal_xy_->at(1));

    const auto goal_xy = pending_goal_xy_.value();
    pending_goal_xy_.reset();

    if (!start_cell.has_value() || !goal_cell.has_value()) {
      has_path_ = false;
      publishEmptyPath();
      RCLCPP_WARN(get_logger(), "Start or goal is outside the planning map.");
      return;
    }

    const MaximinPlanResult result = planMaximinClearancePath(
      grid,
      start_cell.value(),
      goal_cell.value(),
      planner_tolerance_,
      {},
      clearance_target_);

    if (!result.success) {
      has_path_ = false;
      publishEmptyPath();
      RCLCPP_WARN(get_logger(), "%s", result.message.c_str());
      return;
    }

    last_path_ = pathFromCells(grid, result.cells, goal_cell.value(), goal_xy);
    has_path_ = true;
    path_pub_->publish(last_path_);
    RCLCPP_INFO(get_logger(), "Planned path with %zu poses.", last_path_.poses.size());
  }

  nav_msgs::msg::Path pathFromCells(
    const GridSpec & grid,
    const std::vector<GridCell> & cells,
    const GridCell & requested_goal_cell,
    const std::array<double, 2> & requested_goal_xy)
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_id_;
    path.header.stamp = now();
    path.poses.reserve(cells.size());

    for (std::size_t index = 0; index < cells.size(); ++index) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      const auto world = mapToWorld(grid, cells[index]);
      pose.pose.position.x = world.first;
      pose.pose.position.y = world.second;
      pose.pose.position.z = 0.0;
      if (index + 1 == cells.size() && cells[index] == requested_goal_cell) {
        pose.pose.position.x = requested_goal_xy[0];
        pose.pose.position.y = requested_goal_xy[1];
      }
      pose.pose.orientation = orientationForPathPoint(grid, cells, index);
      path.poses.push_back(pose);
    }

    return path;
  }

  void publishEmptyPath()
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_id_;
    path.header.stamp = now();
    path_pub_->publish(path);
  }

  std::string frame_id_;
  std::string map_topic_;
  std::string goal_topic_;
  std::string odom_topic_;
  std::string path_topic_;
  double path_republish_frequency_{1.0};
  double robot_radius_{0.4};
  double inflation_margin_{0.05};
  int64_t map_occupied_threshold_{50};
  bool treat_unknown_as_occupied_{true};
  double planner_tolerance_{0.0};
  double clearance_target_{0.60};

  std::optional<GridSpec> grid_;
  std::optional<std::array<double, 2>> robot_xy_;
  std::optional<std::array<double, 2>> pending_goal_xy_;
  nav_msgs::msg::Path last_path_;
  bool has_path_{false};
  bool reported_map_yaw_{false};

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr republish_timer_;
};

}  // namespace go2_nav2

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<go2_nav2::MaximinAStarPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
