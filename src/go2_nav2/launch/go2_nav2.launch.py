import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_share = get_package_share_directory('go2_nav2')

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    global_frame = LaunchConfiguration('global_frame')
    robot_base_frame = LaunchConfiguration('robot_base_frame')
    cloud_topic = LaunchConfiguration('cloud_topic')
    scan_topic = LaunchConfiguration('scan_topic')
    goal_topic = LaunchConfiguration('goal_topic')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')

    cmd_vel_nav_topic = '/cmd_vel_nav'
    lifecycle_nodes = [
        'planner_server',
        'controller_server',
        'behavior_server',
        'bt_navigator',
        'velocity_smoother',
    ]

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            param_rewrites={
                'bt_navigator.ros__parameters.global_frame': global_frame,
                'bt_navigator.ros__parameters.robot_base_frame': robot_base_frame,
                'local_costmap.local_costmap.ros__parameters.global_frame': global_frame,
                'local_costmap.local_costmap.ros__parameters.robot_base_frame': robot_base_frame,
                'local_costmap.local_costmap.ros__parameters.obstacle_layer.scan.topic':
                    scan_topic,
                'global_costmap.global_costmap.ros__parameters.global_frame': global_frame,
                'global_costmap.global_costmap.ros__parameters.robot_base_frame':
                    robot_base_frame,
                'behavior_server.ros__parameters.local_frame': global_frame,
                'behavior_server.ros__parameters.global_frame': global_frame,
                'behavior_server.ros__parameters.robot_base_frame': robot_base_frame,
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg_share, 'config', 'go2_nav2_params.yaml'),
            description='Full path to the ROS 2 parameters file.',
        ),
        DeclareLaunchArgument(
            'global_frame',
            default_value='odom',
            description='Nav2 global frame. The Dreamer map is published in odom.',
        ),
        DeclareLaunchArgument(
            'robot_base_frame',
            default_value='base',
            description='Robot base TF frame. Override to base_link if the simulator uses base_link.',
        ),
        DeclareLaunchArgument(
            'cloud_topic',
            default_value='/cloud',
            description='Input point cloud topic to project into a laser scan.',
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='Laser scan topic produced from the point cloud.',
        ),
        DeclareLaunchArgument(
            'goal_topic',
            default_value='/goal_pose',
            description='PoseStamped topic accepted by Nav2 BT navigator.',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Final smoothed velocity command topic for go2_controller.',
        ),

        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        SetParameter(name='use_sim_time', value=use_sim_time),

        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            output='screen',
            parameters=[{
                'target_frame': robot_base_frame,
                'transform_tolerance': 0.05,
                'min_height': -0.25,
                'max_height': 1.20,
                'angle_min': -3.14159,
                'angle_max': 3.14159,
                'angle_increment': 0.0087,
                'scan_time': 0.1,
                'range_min': 0.10,
                'range_max': 8.0,
                'use_inf': True,
            }],
            remappings=remappings + [
                ('cloud_in', cloud_topic),
                ('scan', scan_topic),
            ],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[configured_params],
            remappings=remappings,
        ),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[configured_params],
            remappings=remappings + [('cmd_vel', cmd_vel_nav_topic)],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[configured_params],
            remappings=remappings + [('cmd_vel', cmd_vel_nav_topic)],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[configured_params],
            remappings=remappings + [('goal_pose', goal_topic)],
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[configured_params],
            remappings=remappings + [
                ('cmd_vel', cmd_vel_nav_topic),
                ('cmd_vel_smoothed', cmd_vel_topic),
            ],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',
            output='screen',
            parameters=[
                configured_params,
                {
                    'autostart': True,
                    'node_names': lifecycle_nodes,
                },
            ],
        ),
    ])
