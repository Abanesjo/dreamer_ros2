cmake_minimum_required(VERSION 3.8)
project(go2_nav2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(tf2_ros REQUIRED)

set(dependencies
  geometry_msgs
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  tf2_ros
)

add_library(go2_nav2_maximin_astar SHARED
  src/maximin_astar.cpp
  src/maximin_astar_planner.cpp
)
target_include_directories(go2_nav2_maximin_astar
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(go2_nav2_maximin_astar ${dependencies})
pluginlib_export_plugin_description_file(nav2_core maximin_astar_plugin.xml)

add_executable(maximin_astar_planner_node
  src/maximin_astar_planner_node.cpp
)
target_link_libraries(maximin_astar_planner_node
  go2_nav2_maximin_astar
)
ament_target_dependencies(maximin_astar_planner_node ${dependencies})

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES maximin_astar_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS go2_nav2_maximin_astar maximin_astar_planner_node
  EXPORT export_go2_nav2_maximin_astar
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_add_gtest(test_maximin_astar
    test/test_maximin_astar.cpp
  )
  if(TARGET test_maximin_astar)
    target_link_libraries(test_maximin_astar go2_nav2_maximin_astar)
    target_include_directories(test_maximin_astar
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
  endif()

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(go2_nav2_maximin_astar)
ament_export_targets(export_go2_nav2_maximin_astar HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies})

ament_package()
