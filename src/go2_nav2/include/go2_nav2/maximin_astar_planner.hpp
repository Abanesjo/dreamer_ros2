#ifndef GO2_NAV2__MAXIMIN_ASTAR_PLANNER_HPP_
#define GO2_NAV2__MAXIMIN_ASTAR_PLANNER_HPP_

#include <functional>
#include <memory>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2_ros/buffer.h"

namespace go2_nav2
{

class MaximinAStarPlanner : public nav2_core::GlobalPlanner
{
public:
  MaximinAStarPlanner() = default;
  ~MaximinAStarPlanner() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    std::function<bool()> cancel_checker) override;

private:
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string name_;
  std::string global_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  double tolerance_{0.0};
  bool allow_unknown_{false};
  bool use_final_approach_orientation_{false};
};

}  // namespace go2_nav2

#endif  // GO2_NAV2__MAXIMIN_ASTAR_PLANNER_HPP_
