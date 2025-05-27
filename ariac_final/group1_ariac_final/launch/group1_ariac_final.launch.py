import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction
)
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration

from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    # Create parameters dictionary
    parameters = {"use_sim_time": True}
    parameters["use_moveit"] = True

    # Get the operation mode from launch configuration
    operation_mode = LaunchConfiguration("operation_mode").perform(context)
    parameters["operation_mode"] = operation_mode

    urdf = os.path.join(
        get_package_share_directory("ariac_description"),
        "urdf/ariac_robots/ariac_robots.urdf.xacro",
    )

    # RViz config path
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("group1_ariac_final"), "config", "rviz_config.rviz"]
    )

    moveit_config = (
        MoveItConfigsBuilder("ariac_robots", package_name="ariac_moveit_config")
        .robot_description(urdf)
        .robot_description_semantic(file_path="config/ariac_robots.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .joint_limits(file_path="config/joint_limits.yaml")
        .moveit_cpp(
            file_path=get_package_share_directory("group1_ariac_final")
            + "/config/moveit_config.yaml"
        )
        .to_moveit_configs()
    )

    start_rviz = LaunchConfiguration("rviz")
    program_choice = LaunchConfiguration("program")
    

    parameters.update(moveit_config.to_dict())

    
    # Python node that will start directly if RViz is not enabled
    group1_ariac_final_node_py_direct = Node(
        package="group1_ariac_final",
        # executable="ceiling_grab.py",
        executable="floor_grab.py",
        output="screen",
        parameters=[parameters],
        condition=LaunchConfigurationEquals("program", "python") and 
                  UnlessCondition(start_rviz),
    )
    
    # Python node that will start directly if RViz is not enabled
    group1_ariac_final_node_py_ceiling = Node(
        package="group1_ariac_final",
        # executable="ceiling_grab.py",
        executable="ceiling_grab.py",
        output="screen",
        parameters=[parameters],
        condition=LaunchConfigurationEquals("program", "python") and 
                  UnlessCondition(start_rviz),
    )
    
    # Python node that will be started by event handler if RViz is enabled
    group1_ariac_final_node_py_after_rviz = Node(
        package="group1_ariac_final",
        # executable="ceiling_grab.py",
        executable="ceiling_grab.py",        
        output="screen",
        parameters=[parameters],
        condition=LaunchConfigurationEquals("program", "python"),
    )
    
    group1_ariac_final_node_cpp = Node(
        package="group1_ariac_final",
        executable="group1_ariac_final_cpp",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True},
            {"operation_mode": operation_mode}  # Pass operation mode to C++ node
        ],
        arguments=['--ros-args', '--log-level', 'move_group_interface:=warn', '--log-level', 'moveit_trajectory_processing.time_optimal_trajectory_generation:=error'],
        condition=LaunchConfigurationEquals("program", "cpp"),
    )

    # RViz Node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
        condition=IfCondition(start_rviz),
    )

    # Register event handler to start Python node after RViz starts with a 5-second delay
    start_python_after_rviz = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=rviz_node,
            on_start=[
                # Add a TimerAction to delay the start of the Python node
                TimerAction(
                    period=5.0,  # 5-second delay
                    actions=[group1_ariac_final_node_py_after_rviz]
                )
            ],
        ),
        condition=IfCondition(start_rviz) and 
                  LaunchConfigurationEquals("program", "python"),
    )

    # Move Group node
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("ariac_moveit_config"),
                "/launch",
                "/ariac_robots_moveit.launch.py",
            ]
        ),
        condition=IfCondition(str(parameters["use_moveit"])),
    )

    nodes_to_start = [
        group1_ariac_final_node_cpp,
        group1_ariac_final_node_py_direct,  # Will only run if RViz is disabled
        group1_ariac_final_node_py_ceiling,
        move_group,
        rviz_node,
        start_python_after_rviz,  # Will start Python node after RViz if RViz is enabled
    ]

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz",
            default_value="false",
            description="Launch RViz visualization?",
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "program",
            default_value="python",
            description="Choose which program implementation to run. Options: 'python' or 'cpp'",
        )
    )
    
    # Add new argument for operation mode
    declared_arguments.append(
        DeclareLaunchArgument(
            "operation_mode",
            default_value="pick_place_tray",
            description="Choose operation mode. Options: 'pick_place_tray' or 'pick_place_part'",
        )
    )

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )