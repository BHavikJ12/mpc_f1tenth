#!/usr/bin/env python3
"""
Launch file for MPCC controller
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare arguments
    waypoints_file_arg = DeclareLaunchArgument(
        'waypoints_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('mpcc_controller'),
            'config',
            'waypoints.csv'
        ]),
        description='Path to waypoints CSV file'
    )
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('mpcc_controller'),
            'config',
            'mpcc_params.yaml'
        ]),
        description='Path to MPCC parameters YAML file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    visualize_arg = DeclareLaunchArgument(
        'visualize',
        default_value='true',
        description='Publish visualization markers'
    )
    
    # MPCC node
    mpcc_node = Node(
        package='mpcc_controller',
        executable='mpcc_node',
        name='mpcc_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'waypoints_file': LaunchConfiguration('waypoints_file'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'visualize': LaunchConfiguration('visualize'),
            }
        ],
        remappings=[
            ('/odom', '/amcl_pose'),
            ('/scan', '/scan'),
            ('/drive', '/drive'),
        ]
    )
    
    return LaunchDescription([
        waypoints_file_arg,
        params_file_arg,
        use_sim_time_arg,
        visualize_arg,
        mpcc_node,
    ])
