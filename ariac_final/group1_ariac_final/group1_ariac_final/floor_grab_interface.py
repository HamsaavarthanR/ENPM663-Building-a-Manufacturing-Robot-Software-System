#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup


from control_msgs.msg import JointControllerState
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Quaternion, PoseStamped, Pose, Point
from ariac_msgs.msg import CompetitionState
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject, PlanningScene
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, ApplyPlanningScene
from std_msgs.msg import Header



from tf_transformations import quaternion_from_euler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from copy import copy
from math import cos, sin, pi

from ariac_msgs.msg import (
    CompetitionState as CompetitionStateMsg,
    BreakBeamStatus as BreakBeamStatusMsg,
    AdvancedLogicalCameraImage as AdvancedLogicalCameraImageMsg,
    Part as PartMsg,
    PartPose as PartPoseMsg,
    AGVStatus as AGVStatusMsg,
    VacuumGripperState,
    ConveyorParts as ConveyorPartsMsg,
    Order as OrderMsg,
)

from ariac_msgs.srv import (
    VacuumGripperControl,
    ChangeGripper,
    MoveAGV,
    PerformQualityCheck,
    SubmitOrder,
)
from std_srvs.srv import Trigger


from ament_index_python import get_package_share_directory

from moveit import MoveItPy
from moveit.core.robot_state import RobotState, robotStateToRobotStateMsg
from moveit_msgs.srv import ApplyPlanningScene
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.kinematic_constraints import construct_joint_constraint

import time
from os import path
import pyassimp
import math


# Fancy log and utils from lecture9
from group1_ariac_final.fancy_log import FancyLog

from group1_ariac_final.utils import (
    multiply_pose,
    rpy_from_quaternion,
    rad_to_deg_str,
    quaternion_from_euler,
    build_pose,
    AdvancedLogicalCameraImage,
    Order,
    KittingTask,
    CombinedTask,
    AssemblyTask,
    KittingPart,
)

#LOCATOR
from group1_ariac_msgs.msg import BinPartsPoseLot, ConveyorPartsPoseLot
from group1_ariac.retrieve_orders_interface import OrderClass
from ariac_msgs.msg import KitTrayPose, BinParts, ConveyorParts


# NOTE: use `ros2 param get /<controller_name> joints` to get the joint list for a controller topic

class Error(Exception):
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return repr(self.value)


class FloorRobotControl(Node):
    """
    Description: node class for controlling the floor robot
    """
    
    def __init__(self, node_name='floor_robot_node'):
        
        super().__init__(node_name)
        
        # Creating floor robot in moveit
        self._floor_robot = MoveItPy(node_name='floor_robot_moveit_py')                         # robot described in MoveIt
        self._floor_robot_planner = self._floor_robot.get_planning_component("floor_robot")   # planner component
        self._floor_robot_monitor = self._floor_robot.get_planning_scene_monitor()              # scene monitor
        self._floor_robot_state = RobotState(self._floor_robot.get_robot_model())               # current joint state of the floor robot
        
        # status
        self._planning_scene_ready = False      # if everything has been imported into the planning scene
        self._started_moveit = False            # whether moveit has been started by the node yet
        self._competition_state = CompetitionState.IDLE
        self._count = 0             # counter for planning attempts
        
        # planning scene
        self._mesh_file_path = get_package_share_directory("group1_ariac_final") + "/meshes/"
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._world_collision_objects = []
        
        # robot gripper states
        self._floor_robot_gripper_state = None
        self._floor_robot_attached_part = None
        self._pending_gripper_state = None
        self._gripper_state_future = None
        
        # callback groups
        self._reentrant_cb_group = ReentrantCallbackGroup()
        self.ariac_cb_group = MutuallyExclusiveCallbackGroup()
        self.moveit_cb_group = MutuallyExclusiveCallbackGroup()
        self.service_cb_group = ReentrantCallbackGroup()
        
        
        # map gripper state to visible message
        self._gripper_states = {True: "enabled", False: "disabled"}
        
        # AGV tray lock clients
        self._lock_agv_tray_clients = {}
        self._unlock_agv_tray_clients = {}
        self._move_agv_clients = {}
        self._agv_status_listeners = {}
        for i in range(1, 5):
            self._lock_agv_tray_clients[i] = self.create_client(
                Trigger, f"/ariac/agv{i}_lock_tray"
            )
            self._unlock_agv_tray_clients[i] = self.create_client(
                Trigger, f"/ariac/agv{i}_unlock_tray"
            )      
            self._move_agv_clients[i] = self.create_client(
                MoveAGV, f"ariac/move_agv{i}",
                callback_group=self._reentrant_cb_group,
            )
            self._agv_status_listeners[i] = self.create_subscription(
                AGVStatusMsg, f"ariac/agv{i}_status", self._agv_status_cb,10,callback_group=self._reentrant_cb_group
            )
        
        # enable or disable gripper
        self._floor_gripper_enable = self.create_client(
            VacuumGripperControl, "/ariac/floor_robot_enable_gripper"
        )
        

        # Constants for handling parts
        self._pick_offset = 0.003  # Offset for picking up parts
        self._drop_height = 0.01  # Height above tray for part release
        self._kit_tray_thickness = 0.01  # Thickness of the kit tray
        
        # subscriber to vacuum gripper state
        self._floor_robot_gripper_state_sub = self.create_subscription(
            VacuumGripperState,
            "/ariac/floor_robot_gripper_state",
            self._floor_robot_gripper_state_cb,
            qos_profile_sensor_data,
            callback_group=self._reentrant_cb_group,
        )
        
        # allow change of gripper of floor robot
        self._change_gripper_client = self.create_client(
            ChangeGripper,
            "/ariac/floor_robot_change_gripper",
            callback_group=self._reentrant_cb_group,
        )
        
        #allows computing of cartesian path
        self._get_cartesian_path_client = self.create_client(
            GetCartesianPath,
            "compute_cartesian_path",
            callback_group=self.service_cb_group,
        )
        
        # timer for checking the scene
        self.scene_timer = self.create_timer(
            1.0,    # Hz
            self._scene_cb
        )

        # subscriber to the competition state
        self.comp_state_sub = self.create_subscription(
            CompetitionState,
            "/ariac/competition_state",
            self._competition_state_cb,
            10
        )
        
        # create service client for the ariac/submit_order service
        self._submit_client = self.create_client(
            SubmitOrder,                # service type
            "ariac/submit_order",    # service name
            callback_group = self._reentrant_cb_group  # callback
        )
        
        # create service client for the ariac/end_competition/ service
        self._end_competitions_client = self.create_client(
            Trigger,                # service type
            "ariac/end_competition",    # service name
            callback_group = self._reentrant_cb_group # callback
        )
        
        self._rail_positions = {
            "agv1": -4.5,
            "agv2": -1.2,
            "agv3": 1.2,
            "agv4": 4.5,
            "left_bins": 3,
            "right_bins": -3,
        }
        
        # ----------------------------------------------------------------------
        # Data Structures
        # ----------------------------------------------------------------------
        self._world_collision_objects = []

        self._floor_joint_positions_arrs = {
            "floor_kts1_js_": [4.0, 1.57, -1.57, 1.57, -1.57, -1.57, 0.0],
            "floor_kts2_js_": [-4.0, -1.57, -1.57, 1.57, -1.57, -1.57, 0.0],
            "left_bins": [3.0, 0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            "right_bins": [-3.0, 0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            "floor_conveyor_js_": [
                0.0,
                3.14,
                -0.9162979,
                2.04204,
                -2.67035,
                -1.57,
                0.0,
            ],
        }
        for i in range(1, 5):
            self._floor_joint_positions_arrs[f"agv{i}"] = [
                self._rail_positions[f"agv{i}"],
                0.0,
                -1.57,
                1.57,
                -1.57,
                -1.57,
                0.0,
            ]
        self._floor_position_dict = {
            key: self._create_floor_joint_position_state(
                self._floor_joint_positions_arrs[key]
            )
            for key in self._floor_joint_positions_arrs.keys()
        }
        
        self. _part_heights = {
            PartMsg.BATTERY: 0.04,
            PartMsg.PUMP: 0.12,
            PartMsg.REGULATOR: 0.07,
            PartMsg.SENSOR: 0.07,
        }
        
        
        self._part_colors = {
            PartMsg.RED: "red",
            PartMsg.BLUE: "blue",
            PartMsg.GREEN: "green",
            PartMsg.ORANGE: "orange",
            PartMsg.PURPLE: "purple",
        }

        self._part_types = {
            PartMsg.BATTERY: "battery",
            PartMsg.PUMP: "pump",
            PartMsg.REGULATOR: "regulator",
            PartMsg.SENSOR: "sensor",
        }
        
        self._quad_offsets = {
            1: (0.15, 0.15),  # Quadrant 1
            2: (0.15, -0.15),  # Quadrant 2
            3: (-0.15, 0.15),  # Quadrant 3
            4: (-0.15, -0.15),  # Quadrant 4
        }
        
        ##########################################################################################################
        # LOCATOR
        ##########################################################################################################
        
        self.CONVEYOR_VEL_Y = 0.2

        #suscriber to retrieve tray location
        self._tray_location_subscriber = self.create_subscription(KitTrayPose, "group1_ariac/tray_part_poses", self.tray_location_cb, 10,callback_group = self._reentrant_cb_group)
        
        # #suscriber to retrieve conveyor part location
        self._conveyor_location_subscriber = self.create_subscription(ConveyorPartsPoseLot, "group1_ariac/conveyor_part_poses", self.conveyor_location_cb, 10,callback_group = self._reentrant_cb_group)
        
        #suscriber to retrieve bin part location
        self._bin_location_subscriber = self.create_subscription(BinPartsPoseLot, "group1_ariac/bin_part_poses", self.bin_location_cb, 10,callback_group = self._reentrant_cb_group)
        
        # list to store order messages
        self._order_list = []
        
        #subscriber to orders
        self._order_subscriber = self.create_subscription(OrderMsg, "ariac/orders", self.order_storage_cb, 10, callback_group = self._reentrant_cb_group)
        
        # storage for location of parts and trays
        self._part_storage = {}
        self._tray_storage = {}
        
        
        # map part color and type to string
        self.color_mapping = {
            PartMsg.RED    : "red",
            PartMsg.GREEN  : "green",
            PartMsg.BLUE   : "blue",
            PartMsg.ORANGE: "orange",
            PartMsg.PURPLE : "purple"
        }

        self.type_mapping = {
            PartMsg.BATTERY   : "battery",
            PartMsg.PUMP     : "pump",
            PartMsg.SENSOR    : "sensor",
            PartMsg.REGULATOR: "regulator"
        }
        
        
    ##########################################################################################################
    # OTHER
    ##########################################################################################################

    # copied from lecture 9
    def _competition_state_cb(self, msg: CompetitionState):
        """
        Callback for the /ariac/competition_state topic.

        This function processes competition state updates, logging state changes
        and storing the current state for use in decision making.

        Args:
            msg (CompetitionStateMsg): Message containing the current competition state
        """
        # Log if competition state has changed
        if self._competition_state != msg.competition_state:
            self.get_logger().info(
                f"Competition state is: {msg.competition_state}", throttle_duration_sec=1.0
            )

        self._competition_state = msg.competition_state
        
    ##########################################################################################################
    # LOCATOR
    ##########################################################################################################
    
    
    def tray_location_cb(self, msg:KitTrayPose):
        """Retrieve tray id and pose

        Args:
            msg (KitTrayPose): contains tray id and pose
        """
        # retrieve id and pose
        tray_id = msg.id        
        tray_pose = msg.pose
        
        # extract pose position
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # extract pose orientation
        a = msg.pose.orientation.x
        b = msg.pose.orientation.y
        c = msg.pose.orientation.w
        d = msg.pose.orientation.z
        
        # log id and pose
        #self.get_logger().info(f"-Tray {tray_id}: [{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
        
        # only keeps track of one tray id
        if self._tray_storage.get(tray_id) == None:
            self._tray_storage[tray_id] = [x, y, z, a, b, c, d]
            ### need to publish now
        
        return
    
    def tray_location_cb(self, msg:KitTrayPose):
        """Retrieve tray id and pose

        Args:
            msg (KitTrayPose): contains tray id and pose
        """
        # retrieve id and pose
        tray_id = msg.id        
        tray_pose = msg.pose
        
        # extract pose position
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # extract pose orientation
        a = msg.pose.orientation.x
        b = msg.pose.orientation.y
        c = msg.pose.orientation.w
        d = msg.pose.orientation.z
        
        # log id and pose
        #self.get_logger().info(f"-Tray {tray_id}: [{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
        
        # only keeps track of one tray id
        if self._tray_storage.get(tray_id) == None:
            self._tray_storage[tray_id] = [x, y, z, a, b, c, d]
            ### need to publish now
        
        return
    
    def conveyor_location_cb(self, msg:ConveyorPartsPoseLot):
        """Retrieve conveyor parts and pose

        Args:
            msg (ConveyorPartsPoseLot): contains all parts on the conveyor and their pose
        """
        
        # goes through all conveyor parts and retrieve their initial and predicted locations
        for conveyor_part in msg.conveyor_parts:
            
            first_detection = conveyor_part.initial_detection
            
            #current_part_color = self.color_mapping[first_detection.part.color]
            #current_part_type = self.type_mapping[first_detection.part.type]
            current_part_color = first_detection.part.color
            current_part_type = first_detection.part.type
            
            # pose at initial detection
            x = first_detection.pose.position.x
            y = first_detection.pose.position.y
            z = first_detection.pose.position.z
            
            a = first_detection.pose.orientation.x                     
            b = first_detection.pose.orientation.y
            c = first_detection.pose.orientation.z
            d = first_detection.pose.orientation.w
            
            # predictions
            x_p = []
            y_p = []
            z_p = []
            for prediction in conveyor_part.predictions:
                x_p.append(prediction.pose.position.x)
                y_p.append(prediction.pose.position.y)
                z_p.append(prediction.pose.position.z)

            key = str(current_part_color) + " " + str(current_part_type)          
            
            # check if part of same color and type has already been detected before storing
            if self._part_storage.get(key) is None:
                current_part_location_and_pose = {}
                current_part_location_and_pose["location"] = "conveyor"
                current_part_location_and_pose["pose"] = [x, y, z, a, b, c, d]
                current_part_location_and_pose['predictions'] = []
                
                #loop through all predictions and add them to array
                for prediction in conveyor_part.predictions:
                    
                    x = prediction.pose.position.x
                    y = prediction.pose.position.y
                    z = prediction.pose.position.z
                    
                    a = prediction.pose.orientation.x
                    b = prediction.pose.orientation.y
                    c = prediction.pose.orientation.z
                    d = prediction.pose.orientation.w
                    
                    current_part_location_and_pose['predictions'].append([x, y, z, a, b, c, d])
                
                
                self._part_storage[key] = [current_part_location_and_pose]
                
            else:
                # add part to list
                current_part_location_and_pose = {}
                current_part_location_and_pose["location"] = "conveyor"
                current_part_location_and_pose["pose"] = [x, y, z, a, b, c, d]
                
                
                current_part_location_and_pose['predictions'] = []
                    
                # self.get_logger().info(f"{current_part_color} {current_part_type}:") 
                
                #loop through all predictions and add them to array
                for prediction in conveyor_part.predictions:
                    
                    x = prediction.pose.position.x
                    y = prediction.pose.position.y
                    z = prediction.pose.position.z
                    
                    a = prediction.pose.orientation.x
                    b = prediction.pose.orientation.y
                    c = prediction.pose.orientation.z
                    d = prediction.pose.orientation.w
                    
                    current_part_location_and_pose['predictions'].append([x, y, z, a, b, c, d])
                    # self.get_logger().info(f"   -Prediction [{len(current_part_location_and_pose['predictions'])}s]: [{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
                    
                self._part_storage[key].append(current_part_location_and_pose)
                
            # log all parts being tracked in conveyor
            self.get_logger().info(f"{self.color_mapping[current_part_color]} {self.type_mapping[current_part_type]}:")                 
            self.get_logger().info(f"   -Location: conveyor")
            self.get_logger().info(f"   -First detection: [{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
            self.get_logger().info(f"   -Prediction [1s]: [{x + 0.0}, {y + self.CONVEYOR_VEL_Y}, {z + 0.0}] [{a}, {b}, {c}, {d}]")
            self.get_logger().info(f"   -Prediction [2s]: [{x + 0.0}, {y + 2*self.CONVEYOR_VEL_Y}, {z + 0.0}] [{a}, {b}, {c}, {d}]")
        
        return
    
    
    def bin_location_cb(self, msg: BinPartsPoseLot):
        """Retrieve bin parts and pose

        Args:
            msg (BinPartsPoseLot): contains all parts in bins and their pose
        """
        
        # loop through bins and extract parts
        for bin in msg.bins:
             
            bin_number = bin.bin_number
            
            # loop through parts in current bin extract info
            for part_pose in bin.part_poses:
                # process part color and type
                current_part_color = part_pose.part.color
                current_part_type = part_pose.part.type
                
                # process part pose location
                x = part_pose.pose.position.x
                y = part_pose.pose.position.y
                z = part_pose.pose.position.z
                
                # process part pose orientation
                a = part_pose.pose.orientation.x
                b = part_pose.pose.orientation.y
                c = part_pose.pose.orientation.z
                d = part_pose.pose.orientation.w
                
                # storing part to log later                                   
                key = str(current_part_color) + " " + str(current_part_type)
                
                # check if part of same color and type has already been detected before storing
                if self._part_storage.get(key) is None:
                    current_part_location_and_pose = {}
                    current_part_location_and_pose["location"] = bin_number
                    current_part_location_and_pose["pose"] = [x, y, z, a, b, c, d]
                    self._part_storage[key] = [current_part_location_and_pose]
                    
                else:
                    # add part to list
                    current_part_location_and_pose = {}
                    current_part_location_and_pose["location"] = bin_number
                    current_part_location_and_pose["pose"] = [x, y, z, a, b, c, d]
                    self._part_storage[key].append(current_part_location_and_pose)
                    
                # self.get_logger().info(f"{current_part_color} {current_part_type}:")
                # self.get_logger().info(f"   -Location: {bin_number}")
                # self.get_logger().info(f"   -[{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
    
    def order_storage_cb(self, msg:OrderMsg):       
        """convert order to order class and add to list
        """
        self._order_list.append(OrderClass(msg))
    
    
    ##########################################################################################################
    # FLOOR ROBOT
    ##########################################################################################################

    def _test_floor_robot(self):
        
        # if no order has been found do nothing
        while len(self._order_list) == 0:
            pass
        
        # move tray
        current_order = self._order_list[0]
        
        # current order info
        tray_id = current_order._tray_id
        agv_num = current_order._agv_num

        # wait until tray pose is found
        while  self._tray_storage.get(tray_id) is None:
            pass
        
        #extracting pose and orientation
        x, y, z, a, b, c, d = self._tray_storage.get(tray_id)
        
        #creating tray pose (in camera frames)
        tray_pose = Pose()
        tray_pose.position.x = x
        tray_pose.position.y = y
        tray_pose.position.z = z
        tray_pose.orientation = Quaternion(x=a, y=b, z=c, w=d)
        
        # converting to world frame    
        if y < 0:
            station = "kts1"
        else: 
            station = "kts2"
        
        # Change gripper to tray gripper if needed
        if self._floor_robot_gripper_state.type != "tray_gripper":
            gripper_changed = self._floor_robot_change_gripper(station, "trays")
            if not gripper_changed:
                FancyLog.error(self.get_logger(), "Failed to change to tray gripper")
                return False
            
        # ensure that gripper is in correct orientation
        tray_rotation = rpy_from_quaternion(tray_pose.orientation)[2]
        
        q = quaternion_from_euler(0.0, pi, tray_rotation)
        
        
        tray_pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
            
        # Move to pick up tray
        waypoints = []
        gripper_orientation = quaternion_from_euler(0.0, pi, tray_rotation)

        # First move above the tray
        self._move_floor_robot_to_pose(
            build_pose(
                tray_pose.position.x,
                tray_pose.position.y,
                tray_pose.position.z + 0.5,
                gripper_orientation,
            )
        )

        # Move to grasp position
        waypoints = [
            build_pose(
                tray_pose.position.x,
                tray_pose.position.y,
                tray_pose.position.z + self._pick_offset,
                gripper_orientation,
            )
        ]

        self._move_floor_robot_cartesian(waypoints, 0.2, 0.2, False)

        # Enable gripper and wait for attachment
        self._set_floor_robot_gripper_state(True)

        try:
            self._floor_robot_wait_for_attach(30.0, gripper_orientation)
        except Error as e:
            FancyLog.error(self.get_logger(), f"Failed to attach to tray: {str(e)}")

        # Lift tray
        waypoints.clear()
        waypoints.append(
            build_pose(
                tray_pose.position.x,
                tray_pose.position.y,
                tray_pose.position.z + 0.3,
                gripper_orientation,
            )
        )

        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, True)

        # Move to AGV
        agv_tray_pose = self._frame_world_pose(f"agv{agv_num}_tray")
        agv_yaw = rpy_from_quaternion(agv_tray_pose.orientation)[2]
        agv_rotation = quaternion_from_euler(0.0, pi, agv_yaw)

        # Move above AGV
        self._move_floor_robot_to_pose(
            build_pose(
                agv_tray_pose.position.x,
                agv_tray_pose.position.y,
                agv_tray_pose.position.z + 0.5,
                agv_rotation,
            )
        )

        # Lower the arm
        waypoints = [
            build_pose(
                agv_tray_pose.position.x,
                agv_tray_pose.position.y,
                agv_tray_pose.position.z + 0.01,
                agv_rotation,
            )
        ]
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)

        # Release the tray
        self._set_floor_robot_gripper_state(False)

        # Lock tray to AGV
        self._lock_agv_tray(agv_num)

        # Move up
        waypoints.clear()
        waypoints.append(
            build_pose(
                agv_tray_pose.position.x,
                agv_tray_pose.position.y,
                agv_tray_pose.position.z + 0.3,
                quaternion_from_euler(0.0, pi, 0),
            )
        )

        self._move_floor_robot_cartesian(waypoints, 0.2, 0.2, False)

        self.get_logger().info(f"Successfully placed tray on AGV {agv_num}")
        
        # pickup blue sensor first
        part_pose = Pose()
        part_pose.position.x = -2.080000
        part_pose.position.y = 2.445000
        part_pose.position.z = 0.720000990
        
        part_rotation = pi
        
        part_to_pick = PartMsg()
        part_to_pick.color = PartMsg.BLUE
        part_to_pick.type = PartMsg.SENSOR
        
        quadrant = 2
        
        self.pick_place_part_bin_tray(part_rotation, part_pose, part_to_pick, quadrant)
        
        # pickup orange pump
        part_pose = Pose()
        part_pose.position.x = -2.080000
        part_pose.position.y = 2.805000
        part_pose.position.z = 0.719999
        
        part_rotation = pi
        
        part_to_pick = PartMsg()
        part_to_pick.color = PartMsg.ORANGE
        part_to_pick.type = PartMsg.PUMP
    
        
        quadrant = 1
        
        self.pick_place_part_bin_tray(part_rotation, part_pose, part_to_pick, quadrant)
        
        move_request = MoveAGV.Request()
        move_request.location = AGVStatusMsg.WAREHOUSE
        move_future = self._move_agv_clients[2].call_async(move_request)
        
        while not move_future.done():
            pass        

            
        # # set tray pose above tray (needs to be changed automatically)
        # tray_pose = Pose()
        # tray_pose.position.x = -0.870000
        # tray_pose.position.y = -5.840000
        # tray_pose.position.z = 0.734990 + 0.5
        # q = quaternion_from_euler(0.0, pi, pi)
        # tray_pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        
        # self._floor_robot_goto(tray_pose)
        
        # # set tray pose in pick up position
        # tray_pose = Pose()
        # tray_pose.position.x = -0.870000
        # tray_pose.position.y = -5.840000
        # tray_pose.position.z = 0.734990 + self._pick_offset
        # q = quaternion_from_euler(0.0, pi, pi)
        # tray_pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        
        # self._floor_robot_goto(tray_pose)
     
        # # attach tray to floor robot
        # # Enable gripper and wait for attachment
        # self._set_floor_robot_gripper_state(True)
        
        # try:
        #     self._floor_robot_wait_for_attach(30.0, tray_pose.orientation)
        # except Error as e:
        #     FancyLog.error(self.get_logger(), f"Failed to attach to tray: {str(e)}")
        
        
    def pick_place_part_bin_tray(self, part_rotation, part_pose:Pose, part_to_pick:PartMsg, quadrant):
        
        q = quaternion_from_euler(0.0, pi, part_rotation)
        
        if self._floor_robot_gripper_state.type != "part_gripper":
            # Determine which tool changer station to use
            station = "kts1" if part_pose.position.y < 0 else "kts2"

            # Move to the tool changer and change gripper
            self._move_floor_robot_to_joint_position(f"floor_{station}_js_")
            self._floor_robot_change_gripper(station, "parts")
            
            
        self._move_floor_robot_to_joint_position("right_bins")
        
        # rotation of part(need to be changed)
        gripper_orientation = quaternion_from_euler(0.0, pi, part_rotation)
        
        above_pose = build_pose(
            part_pose.position.x,
            part_pose.position.y,
            part_pose.position.z + 0.15,  # Reduced from 0.3 to 0.15
            gripper_orientation,
        )
        
        # move gripper above part
        self._move_floor_robot_to_pose(above_pose)
        
        
        # PICKING PHASE - Getting closer to the part
        self.get_logger().info("Moving to grasp position")
        waypoints = [
            build_pose(
                part_pose.position.x,
                part_pose.position.y,
                # Optimize height for better first-attempt success
                part_pose.position.z
                + self._part_heights[part_to_pick.type]
                + 0.005,  # You can adjust this value
                gripper_orientation,
            )
        ]
        
        # move_above part
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, False)

        # Enable gripper with less waiting
        self._set_floor_robot_gripper_state(True)
        
        try:
            self._floor_robot_wait_for_attach(
                10.0, gripper_orientation
            )  # Reduced timeout
        except Error as e:
            self.get_logger().error(f"Attachment failed: {str(e)}")

            # Quick recovery
            waypoints = [
                build_pose(
                    part_pose.position.x,
                    part_pose.position.y,
                    part_pose.position.z + 0.2,
                    gripper_orientation,
                )
            ]
            self._move_floor_robot_cartesian(waypoints, 0.5, 0.5, False)
            self._set_floor_robot_gripper_state(False)
            return False
        
        # RETREAT PHASE - Faster retreat
        # Quick lift to clear obstacles
        waypoints = [
            build_pose(
                part_pose.position.x,
                part_pose.position.y,
                part_pose.position.z + 0.2,
                gripper_orientation,
            )
        ]
        self._move_floor_robot_cartesian(waypoints, 0.2, 0.2, False)

        # Return to bin position
        self._move_floor_robot_to_joint_position("right_bins")
        

        # SCENE UPDATE PHASE - Minimal planning scene updates
        # Just record the attached part internally for tracking
        self._floor_robot_attached_part = part_to_pick

        # Only update planning scene if needed
        self._attach_model_to_floor_gripper(part_to_pick, part_pose)
        
        # Place on quadrant ()
        
        self.get_logger().info(f"Placing part in quadrant {quadrant}")

        success = self._floor_robot_place_part_on_kit_tray(2, quadrant)

        if success:
            self.get_logger().info("Successfully placed part on tray")
        else:
            self.get_logger().error("Failed to place part on tray")
            
        
            
    
    def _create_floor_plan(self):
        self.get_logger().info("Planning floor robot trajectory")
        time_start = time.time()
        
        # assume no special parameters for now
        plan = self._floor_robot_planner.plan()
        
        self.get_logger().info(f"---> Total plan time: {(time.time() - time_start):.3f} s")

        if plan:
            self.get_logger().info("---> Planning sucessful")
            return plan
        
        self.get_logger().warn("---> Planning failed")
        return None
    
    def _execute_plan(self, plan):
        self.get_logger().info("Executing plan")
        time_start = time.time()

        # update the simulation with the step in the plan
        with self._floor_robot_monitor.read_write() as scene:
            scene.current_state.update(True)
            self._floor_robot_state = scene.current_state
            trajectory = plan.trajectory

        self._floor_robot.execute(
            trajectory,
            controllers=["floor_robot_controller", "linear_rail_controller"]
        )

        self.get_logger().info(f"---> Total execution time: {(time.time() - time_start):.3f} s")

    def _floor_robot_goto(self, pose : Pose):

        with self._floor_robot_monitor.read_write() as scene:
            self._floor_robot_planner.set_start_state(robot_state=scene.current_state)

            # create the pose message
            target_pose = PoseStamped()
            target_pose.header.frame_id = "world"
            target_pose.pose = pose
            self._floor_robot_planner.set_goal_state(
                pose_stamped_msg=target_pose,
                pose_link="floor_gripper"
            )
        
        # no infinite looping
        while self._count < 3:
            plan = self._create_floor_plan()
            if plan is not None:
                self._execute_plan(plan)
                return True
            else:
                self.get_logger().warn(f"---> Failed attempt ({self._count} of {3})")
                self._count += 1
         
        self.get_logger().error("---> Max attempts reached, failed to move floor robot")
        return False
    
    # determine when agv reach warehouse
    def _agv_status_cb(self, msg:AGVStatusMsg):
        
        if msg.location == AGVStatusMsg.WAREHOUSE:
            # call service
            submit_request = SubmitOrder.Request()
            
            # add order id to be completed
            submit_request.order_id  = "GOODLUCK"
            
            # wait until the service exists before trying to send a request
            while not self._submit_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info("Waiting for submit order service to start...")
                
            self._submit_client.call(submit_request)
            self.get_logger().info(f"Agv arrived at warehouse, order {submit_request.order_id} submitted")
            
            self.end_competition()
        else:
            pass
        
    def end_competition(self):
        """ 
        Class to safely end the competition
        """
        
        # Sending service request to end competition
        end_competition_request = Trigger.Request()
        self._end_competitions_client.call(end_competition_request)
        self.get_logger().info("Ending Competition - All orders submitted and completed.")
        
    
    
    ##########################################################################################################
    # MOVEIT
    ##########################################################################################################

    # adds objects to the scene if needed and then waits for the competition to be started before starting moveit
    # parts borrowed from lecture 9
    def _scene_cb(self):
        
        if not self._started_moveit:
            if not self._planning_scene_ready:
                self.get_logger().info("Creating planning scene")
                ret = self._add_models_to_planning_scene()

                if ret:
                    self.get_logger().info("---> All objects added")
                    self._planning_scene_ready = True
                else:
                    self.get_logger().warn("---> Failed to add all objects, will reattempt on next timer cycle")
                    return
            
            # Only proceed if competition is in the correct state
            if (
                self._competition_state
                in [
                    CompetitionState.STARTED,
                    CompetitionState.ORDER_ANNOUNCEMENTS_DONE,
                ]
            ):
                time.sleep(3.0) # wait a bit before starting any planning
                self.get_logger().info("\n*********")
                self.get_logger().info("*** NOTE: this is where we would start whatever our task is, I'm testing the floor robot movement for now, but that should be replaced eventually")
                self.get_logger().info("\n*********")
                self._test_floor_robot()
                self._started_moveit = True

    # add a specific mesh to the planning scene
    # copied from lecture 9
    def _add_model_to_planning_scene(
        self, name: str, mesh_file: str, model_pose: Pose, frame_id="world"
    ) -> bool:
        """
        Add a mesh model to the planning scene using the PlanningSceneMonitor.

        This function creates a collision object from a mesh file and adds it
        to the planning scene for collision checking during motion planning.

        Args:
            name (str): Unique identifier for the collision object
            mesh_file (str): File name of the mesh in the meshes directory
            model_pose (Pose): Position and orientation of the object
            frame_id (str, optional): Reference frame for the object. Defaults to "world".

        Returns:
            bool: True if the object was successfully added, False otherwise
        """
        self.get_logger().info(f"Adding model {name} to planning scene")

        try:
            # Get the full path to the mesh file
            model_path = self._mesh_file_path + mesh_file

            # Check if the mesh file exists
            if not path.exists(model_path):
                self.get_logger().error(f"Mesh file not found: {model_path}")
                return False

            # Create collision object
            collision_object = self._make_mesh(name, model_pose, model_path, frame_id)

            if collision_object is None:
                self.get_logger().error(f"Failed to create collision object for {name}")
                return False

            # Add to planning scene using the monitor
            with self._floor_robot_monitor.read_write() as scene:
                # Apply the collision object
                scene.apply_collision_object(collision_object)

                # Update the scene
                scene.current_state.update()

            # Add to our tracking list for later reference
            self._world_collision_objects.append(collision_object)

            self.get_logger().info(f"Successfully added {name} to planning scene")
            return True

        except Exception as e:
            self.get_logger().info(f"Error adding model {name} to planning scene: {str(e)}")
            return False

    # adds the meshes to the planning scene so that moveit is aware of possible collisions
    # function copied from lecture 9
    def _add_models_to_planning_scene(self):
        """
        Add collision models to the MoveIt planning scene.

        Populates the planning scene with collision objects for:
        - Bins (bins 1-8)
        - Assembly stations (AS1-AS4)
        - Assembly station briefcases/inserts
        - Conveyor belt
        - Kit tray tables (KTS1 and KTS2)

        These collision objects enable path planning with collision avoidance.
        The function aggregates success status across all object additions.

        Returns:
            bool: True if all objects were added successfully, False otherwise
        """
        self.get_logger().info("Initializing planning scene with collision objects")

        # Start with success as True and maintain it only if all operations succeed
        success = True

        # Add bins
        bin_positions = {
            "bin1": (-1.9, 3.375),
            "bin2": (-1.9, 2.625),
            "bin3": (-2.65, 2.625),
            "bin4": (-2.65, 3.375),
            "bin5": (-1.9, -3.375),
            "bin6": (-1.9, -2.625),
            "bin7": (-2.65, -2.625),
            "bin8": (-2.65, -3.375),
        }

        bin_pose = Pose()
        for bin_name, position in bin_positions.items():
            bin_pose.position.x = position[0]
            bin_pose.position.y = position[1]
            bin_pose.position.z = 0.0
            q = quaternion_from_euler(0.0, 0.0, 3.14159)
            bin_pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)

            # Aggregate success status
            success = success and self._add_model_to_planning_scene(
                bin_name, "bin.stl", bin_pose
            )

        # Add assembly stations
        assembly_station_positions = {
            "as1": (-7.3, 3.0),
            "as2": (-12.3, 3.0),
            "as3": (-7.3, -3.0),
            "as4": (-12.3, -3.0),
        }

        assembly_station_pose = Pose()
        for station_name, position in assembly_station_positions.items():
            assembly_station_pose.position.x = position[0]
            assembly_station_pose.position.y = position[1]
            assembly_station_pose.position.z = 0.0
            q = quaternion_from_euler(0.0, 0.0, 0.0)
            assembly_station_pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)

            # Aggregate success status
            success = success and self._add_model_to_planning_scene(
                station_name, "assembly_station.stl", assembly_station_pose
            )

        # Add assembly briefcases
        assembly_inserts = {
            "as1_insert": "as1_insert_frame",
            "as2_insert": "as2_insert_frame",
            "as3_insert": "as3_insert_frame",
            "as4_insert": "as4_insert_frame",
        }

        for insert_name, frame_id in assembly_inserts.items():
            try:
                insert_pose = self._frame_world_pose(frame_id)
                insert_success = self._add_model_to_planning_scene(
                    insert_name, "assembly_insert.stl", insert_pose
                )
                # Aggregate success status
                success = success and insert_success
            except Exception as e:
                self.get_logger().warn(f"Failed to add assembly insert {insert_name}: {e}")
                # Mark failure but continue with other objects
                success = False

        # Add conveyor
        conveyor_pose = Pose()
        conveyor_pose.position.x = -0.6
        conveyor_pose.position.y = 0.0
        conveyor_pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, 0.0) 
        conveyor_pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)

        # Aggregate success status
        success = success and self._add_model_to_planning_scene(
            "conveyor", "conveyor.stl", conveyor_pose
        )

        # Add kit tray tables
        kts1_table_pose = Pose()
        kts1_table_pose.position.x = -1.3
        kts1_table_pose.position.y = -5.84
        kts1_table_pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, 3.14159)
        kts1_table_pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)

        # Aggregate success status
        success = success and self._add_model_to_planning_scene(
            "kts1_table", "kit_tray_table.stl", kts1_table_pose
        )

        kts2_table_pose = Pose()
        kts2_table_pose.position.x = -1.3
        kts2_table_pose.position.y = 5.84
        kts2_table_pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, 0.0)
        kts2_table_pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)

        # Aggregate success status
        success = success and self._add_model_to_planning_scene(
            "kts2_table", "kit_tray_table.stl", kts2_table_pose
        )

        if success:
            self.get_logger().info("Planning scene initialization complete")
            self._refresh_planning_scene_display()
            # time.sleep(0.5)
        else:
            self.get_logger().warn("Planning scene initialization incomplete - some objects failed to load")

        return success

    # get the pose of the world frame
    # copied from lecture 9
    def _frame_world_pose(self, frame_id: str):
        """
        Get the pose of a frame in the world frame using TF2.

        This function uses the TF2 library to look up the transform between
        the world frame and the specified frame, converting it to a Pose message.

        Args:
            frame_id (str): Target frame ID to get pose for

        Returns:
            Pose: Pose of the frame in the world frame

        Raises:
            Exception: If the transform lookup fails
        """
        self.get_logger().info(f"Getting transform for frame: {frame_id}")

        # Wait for transform to be available
        try:
            # First check if the transform is available
            if not self._tf_buffer.can_transform("world", frame_id, rclpy.time.Time()):
                self.get_logger().warn(
                    f"Transform from world to {frame_id} not immediately available, waiting..."
                )

            # Wait synchronously for the transform
            timeout = rclpy.duration.Duration(seconds=2.0)
            t = self._tf_buffer.lookup_transform(
                "world", frame_id, rclpy.time.Time(), timeout
            )

        except Exception as e:
            self.get_logger().error(f"Failed to get transform for {frame_id}: {str(e)}")
            raise

        pose = Pose()
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = t.transform.rotation

        return pose
    
    # create the mesh in the scene
    # copied from lecture 9
    def _make_mesh(self, name, pose, filename, frame_id) -> CollisionObject:
        """
        Create a collision object from a mesh file.

        This function loads a mesh file using pyassimp and creates a CollisionObject
        message containing the mesh for use in the planning scene.

        Args:
            name (str): Unique identifier for the collision object
            pose (Pose): Position and orientation of the object
            filename (str): File path to the mesh file
            frame_id (str): Reference frame for the object

        Returns:
            CollisionObject: The created collision object ready for planning scene addition

        Raises:
            AssertionError: If no meshes are found in the file
        """
        with pyassimp.load(filename) as scene:
            assert len(scene.meshes), "No meshes found in the file"

            mesh = Mesh()
            for face in scene.meshes[0].faces:
                triangle = MeshTriangle()
                if hasattr(face, "indices"):
                    if len(face.indices) == 3:
                        triangle.vertex_indices = [
                            face.indices[0],
                            face.indices[1],
                            face.indices[2],
                        ]
                        mesh.triangles.append(triangle)
                else:
                    if len(face) == 3:
                        triangle.vertex_indices = [face[0], face[1], face[2]]
                        mesh.triangles.append(triangle)

            for vertex in scene.meshes[0].vertices:
                point = Point()
                point.x = float(vertex[0])
                point.y = float(vertex[1])
                point.z = float(vertex[2])
                mesh.vertices.append(point)

            o = CollisionObject()
            o.header.frame_id = frame_id
            o.id = name
            o.meshes.append(mesh)
            o.mesh_poses.append(pose)
            o.operation = o.ADD

            return o

    def _apply_planning_scene(self, scene):
        """
        Apply a planning scene with robust timeout handling.

        This function sends a planning scene to the ApplyPlanningScene service
        with optimized timeout handling for more responsive operation.

        Args:
            scene (PlanningScene): The planning scene to apply

        Returns:
            bool: True if the planning scene was successfully applied, False otherwise
        """
        try:
            # Create a client for the service
            apply_planning_scene_client = self.create_client(
                ApplyPlanningScene, "/apply_planning_scene"
            )

            # Wait for service to be available with reduced timeout
            if not apply_planning_scene_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("'/apply_planning_scene' service not available")
                return False

            # Create and send the request
            request = ApplyPlanningScene.Request()
            request.scene = scene

            # Send the request
            future = apply_planning_scene_client.call_async(request)

            # Wait for the response with shorter timeout
            timeout_sec = 1.0  # Reduced timeout
            start_time = time.time()

            while not future.done():
                time.sleep(0.01)  # Shorter sleep interval

                if time.time() - start_time > timeout_sec:
                    self.get_logger().warn(
                        "Timeout waiting for planning scene service, trying direct publish instead"
                    )
                    # Try direct publishing as a fallback
                    self._direct_publish_planning_scene(scene)
                    return False  # Assume success with direct publishing

            # Process the result
            result = future.result()
            if result.success:
                self.get_logger().info("Successfully applied planning scene")
                return True
            else:
                self.get_logger().warn("Failed to apply planning scene via service")
                # Try direct publishing as a fallback
                # self._direct_publish_planning_scene(scene)
                return True

        except Exception as e:
            self.get_logger().error(f"Error applying planning scene: {str(e)}")
            return False

    def _direct_publish_planning_scene(self, scene):
        """
        Publish planning scene directly to the planning scene topic.

        This function bypasses the service-based planning scene application
        and publishes directly to the topic for faster operation or as a fallback.

        Args:
            scene (PlanningScene): The planning scene to publish
        """
        if not hasattr(self, "_planning_scene_publisher"):
            self._planning_scene_publisher = self.create_publisher(
                PlanningScene, "/planning_scene", 10
            )
            # Short delay to allow publisher to initialize
            time.sleep(0.1)

        # Set is_diff flag for proper scene update
        scene.is_diff = True

        # Publish the scene
        self._planning_scene_publisher.publish(scene)

        # Give some time for the planning scene to be processed
        time.sleep(0.1)

        self.get_logger().info(
            f"Directly published planning scene with {len(scene.world.collision_objects)} objects"
        )

    def _refresh_planning_scene_display(self):
        """
        Force a refresh of the planning scene display in RViz.

        This function creates and applies an empty differential planning scene
        to trigger a visual update of the planning scene in visualization tools.
        """
        # Create an empty diff planning scene just to trigger a display update
        refresh_scene = PlanningScene()
        refresh_scene.is_diff = True
        self._apply_planning_scene(refresh_scene)
        
        
    def _move_floor_robot_cartesian(
        self, waypoints, velocity, acceleration, avoid_collision=True
    ):
        """
        Move the floor robot along a Cartesian path with optimized speed.

        This function plans and executes a Cartesian path through the specified
        waypoints, applying velocity and acceleration scaling factors for
        performance optimization.

        Args:
            waypoints (list): List of Pose objects defining the path
            velocity (float): Maximum velocity scaling factor (0.0-1.0)
            acceleration (float): Maximum acceleration scaling factor (0.0-1.0)
            avoid_collision (bool, optional): Whether to avoid collisions during
                                              path planning. Defaults to True.

        Returns:
            bool: True if path execution succeeded, False otherwise
        """
        # Increase default velocity and acceleration for faster movement
        velocity = max(0.5, velocity)  # Minimum velocity scaling of 0.5
        acceleration = max(0.5, acceleration)  # Minimum acceleration scaling of 0.5

        # Get the trajectory
        trajectory_msg = self._call_get_cartesian_path(
            waypoints, velocity, acceleration, avoid_collision, "floor_robot"
        )

        if trajectory_msg is None:
            self.get_logger().error("Failed to compute cartesian path")
            return False

        # Execute the trajectory
        with self._floor_robot_monitor.read_write() as scene:
            trajectory = RobotTrajectory(self._floor_robot.get_robot_model())
            trajectory.set_robot_trajectory_msg(scene.current_state, trajectory_msg)
            trajectory.joint_model_group_name = "floor_robot"
            scene.current_state.update(True)
            self._floor_robot_state = scene.current_state

        try:
            self._floor_robot.execute(trajectory, controllers=[])
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to execute cartesian path: {str(e)}")
            return False
        
        
    def _call_get_cartesian_path(
        self,
        waypoints: list,
        max_velocity_scaling_factor: float,
        max_acceleration_scaling_factor: float,
        avoid_collision: bool,
        robot: str,
    ):
        """
        Call the compute_cartesian_path service to generate a Cartesian trajectory.

        This function creates and sends a request to the GetCartesianPath service,
        configuring path constraints and parameters as specified.

        Args:
            waypoints (list): List of Pose objects defining the path
            max_velocity_scaling_factor (float): Maximum velocity scaling (0.0-1.0)
            max_acceleration_scaling_factor (float): Maximum acceleration scaling (0.0-1.0)
            avoid_collision (bool): Whether to avoid collisions during planning
            robot (str): Robot name to plan for (e.g., "floor_robot")

        Returns:
            trajectory_msg: The computed trajectory message, or None if planning failed
        """
        self.get_logger().debug(
            "Getting cartesian path"
        )  # Use debug level for less logging

        request = GetCartesianPath.Request()

        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        request.header = header
        with self._floor_robot_monitor.read_write() as scene:
            request.start_state = robotStateToRobotStateMsg(scene.current_state)

        if robot == "floor_robot":
            request.group_name = "floor_robot"
            request.link_name = "floor_gripper"

        # Always use higher velocity values for faster motion
        request.waypoints = waypoints
        request.max_step = 0.1
        request.avoid_collisions = avoid_collision
        # Override with higher values for faster motion
        request.max_velocity_scaling_factor = max(0.7, max_velocity_scaling_factor)
        request.max_acceleration_scaling_factor = max(
            0.6, max_acceleration_scaling_factor
        )

        future = self._get_cartesian_path_client.call_async(request)

        # Use more efficient waiting with timeout
        timeout = 1.0  # 1 second timeout
        start_time = time.time()
        while not future.done():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.get_logger().warn("Cartesian path planning timeout!")
                return None
            time.sleep(0.01)  # Short sleep to reduce CPU usage

        result: GetCartesianPath.Response
        result = future.result()

        if result.fraction < 0.9:
            self.get_logger().warn(
                f"Path planning incomplete: only {result.fraction:.2f} coverage"
            )

        return result.solution
        
    ##########################################################################################################
    # GRIPPER
    ##########################################################################################################
    
    def _set_floor_robot_gripper_state(self, state):
        """
        Control the floor robot gripper and update planning scene when detaching objects.

        This function sends a service request to enable or disable the vacuum gripper.
        When disabling the gripper, it also updates the planning scene to detach any
        attached parts.

        Args:
            state (bool): True to enable the gripper, False to disable

        Returns:
            Future: The Future object for the service call, or None if the gripper
                    is already in the requested state
        """
        if self._floor_robot_gripper_state.enabled == state:
            self.get_logger().debug(f"Gripper is already {self._gripper_states[state]}")
            return None

        # If disabling the gripper and we have an attached part, detach it in the planning scene
        if not state and self._floor_robot_attached_part is not None:
            part_name = (
                self._part_colors[self._floor_robot_attached_part.color]
                + "_"
                + self._part_types[self._floor_robot_attached_part.type]
            )
            self._detach_object_from_floor_gripper(part_name)
            # Clear the attached part reference
            self._floor_robot_attached_part = None

        request = VacuumGripperControl.Request()
        request.enable = state

        # Store state for use in callback
        self._pending_gripper_state = state

        # Log at debug level instead of info
        self.get_logger().debug(
            f"Changing gripper state to {self._gripper_states[state]}"
        )

        # Use call_async with a callback
        future = self._floor_gripper_enable.call_async(request)
        future.add_done_callback(self._gripper_state_callback)

        # Store and return the future
        self._gripper_state_future = future
        return future
    
    def _floor_robot_gripper_state_cb(self, msg: VacuumGripperState):
        """
        Callback for the /ariac/floor_robot_gripper_state topic.

        This function processes gripper state updates, storing the current
        gripper state for use in decision making and planning.

        Args:
            msg (VacuumGripperState): Message containing the current gripper state
        """
        self._floor_robot_gripper_state = msg
        
    
    def _floor_robot_wait_for_attach(self, timeout: float, orientation: Quaternion):
        """
        Wait for a part to attach to the gripper, making small downward movements if needed.

        This function implements an adaptive approach to part attachment:
        1. First waits briefly to see if the part attaches immediately
        2. If not, makes small incremental downward movements
        3. Continues until attachment is detected or timeout is reached

        Args:
            timeout (float): Maximum time in seconds to wait for attachment
            orientation (Quaternion): Orientation to maintain during movements

        Returns:
            bool: True if part attached successfully

        Raises:
            Error: If timeout is reached or attachment fails after max retries
        """
        with self._floor_robot_monitor.read_write() as scene:
            current_pose = scene.current_state.get_pose("floor_gripper")

        start_time = time.time()
        retry_count = 0

        # First try waiting a short time for attachment without moving
        time.sleep(0.1)
        if self._floor_robot_gripper_state.attached:
            self.get_logger().info("Part attached on first attempt")
            return True

        # while not self._floor_robot_gripper_state.attached and retry_count < max_retries:
        while not self._floor_robot_gripper_state.attached:
            # Move down in larger increments for faster operation
            # z_offset = -0.002 * (retry_count + 1)  # Progressive larger movements
            z_offset = -0.001

            current_pose = build_pose(
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z + z_offset,
                orientation,
            )

            waypoints = [current_pose]
            self._move_floor_robot_cartesian(waypoints, 0.1, 0.1, False)

            # Check if attached after movement
            time.sleep(0.4)  # Short wait

            retry_count += 1

            if time.time() - start_time >= timeout:
                self.get_logger().error("Unable to pick up part: timeout")
                raise Error("Gripper attachment timeout")

        if not self._floor_robot_gripper_state.attached:
            self.get_logger().error("Unable to pick up part: max retries reached")
            raise Error("Gripper attachment failed after max retries")

        self.get_logger().info(f"Part attached after {retry_count} attempts")
        return True
    
    
    def _floor_robot_change_gripper(self, station: str, gripper_type: str):
        """
        Change the gripper on the floor robot.

        This function implements the complete gripper changing process:
        1. Gets the pose of the tool changer frame
        2. Moves the robot to the tool changer station
        3. Calls the gripper change service
        4. Moves away from the tool changer

        Args:
            station (str): Station to change gripper at ("kts1" or "kts2")
            gripper_type (str): Type of gripper to change to ("trays" or "parts")

        Returns:
            bool: True if gripper change succeeded, False otherwise
        """
        FancyLog.info(self.get_logger(), f"Changing gripper to type: {gripper_type}")

        try:
            # Get the pose of the tool changer frame
            tc_pose = self._frame_world_pose(
                f"{station}_tool_changer_{gripper_type}_frame"
            )
        except Exception as e:
            FancyLog.error(
                self.get_logger(), f"Failed to get tool changer frame pose: {str(e)}"
            )
            return False

        # Move above the tool changer
        self._move_floor_robot_to_pose(
            build_pose(
                tc_pose.position.x,
                tc_pose.position.y,
                tc_pose.position.z + 0.7,
                quaternion_from_euler(0.0, pi, 0.0),
            )
        )

        # Move to the tool changer
        waypoints = [
            build_pose(
                tc_pose.position.x,
                tc_pose.position.y,
                tc_pose.position.z,
                quaternion_from_euler(0.0, pi, 0.0),
            )
        ]

        self._move_floor_robot_cartesian(waypoints, 0.2, 0.2, False)

        # Create and send the service request
        request = ChangeGripper.Request()

        if gripper_type == "trays":
            request.gripper_type = ChangeGripper.Request.TRAY_GRIPPER
        elif gripper_type == "parts":
            request.gripper_type = ChangeGripper.Request.PART_GRIPPER

        # Check if service is available with timeout
        if not self._change_gripper_client.wait_for_service(timeout_sec=2.0):
            FancyLog.error(self.get_logger(), "Change gripper service not available")
            return False

        future = self._change_gripper_client.call_async(request)

        # Wait for the response with timeout (5 seconds)
        timeout_sec = 5.0
        start_time = time.time()

        while not future.done():
            if time.time() - start_time > timeout_sec:
                FancyLog.error(
                    self.get_logger(), "Timeout waiting for change gripper service"
                )
                # Move away from the tool changer to avoid being stuck
                waypoints = [
                    build_pose(
                        tc_pose.position.x,
                        tc_pose.position.y,
                        tc_pose.position.z + 0.4,
                        quaternion_from_euler(0.0, pi, 0.0),
                    )
                ]
                self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, False)
                return False

            # Sleep briefly to avoid CPU spinning
            time.sleep(0.05)

        # Check the result
        try:
            result = future.result()
            if not result.success:
                FancyLog.error(
                    self.get_logger(),
                    "Error from change gripper service: "
                    + (
                        result.message
                        if hasattr(result, "message")
                        else "No error message"
                    ),
                )
                return False
        except Exception as e:
            FancyLog.error(
                self.get_logger(),
                f"Exception getting result from change gripper service: {str(e)}",
            )
            return False

        # Move away from the tool changer
        waypoints = [
            build_pose(
                tc_pose.position.x,
                tc_pose.position.y,
                tc_pose.position.z + 0.4,
                quaternion_from_euler(0.0, pi, 0.0),
            )
        ]

        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, False)
        FancyLog.info(
            self.get_logger(), f"Successfully changed to {gripper_type} gripper"
        )
        return True
    
    def _gripper_state_callback(self, future):
        """
        Callback for gripper state change service response.

        This function processes the result of the asynchronous gripper control
        service call, logging appropriate messages based on the result.

        Args:
            future (rclpy.task.Future): The Future object containing the service response
        """
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(
                    f"Changed gripper state to {self._gripper_states[self._pending_gripper_state]}"
                )
            else:
                self.get_logger().warn("Unable to change gripper state")
        except Exception as e:
            self.get_logger().error(f"Gripper state service call failed: {e}")
    
    
    ##########################################################################################################
    # UTILS
    ##########################################################################################################
    
    
    def _frame_world_pose(self, frame_id: str):
        """
        Get the pose of a frame in the world frame using TF2.

        This function uses the TF2 library to look up the transform between
        the world frame and the specified frame, converting it to a Pose message.

        Args:
            frame_id (str): Target frame ID to get pose for

        Returns:
            Pose: Pose of the frame in the world frame

        Raises:
            Exception: If the transform lookup fails
        """
        self.get_logger().info(f"Getting transform for frame: {frame_id}")

        # Wait for transform to be available
        try:
            # First check if the transform is available
            if not self._tf_buffer.can_transform("world", frame_id, rclpy.time.Time()):
                self.get_logger().warn(
                    f"Transform from world to {frame_id} not immediately available, waiting..."
                )

            # Wait synchronously for the transform
            timeout = rclpy.duration.Duration(seconds=2.0)
            t = self._tf_buffer.lookup_transform(
                "world", frame_id, rclpy.time.Time(), timeout
            )

        except Exception as e:
            self.get_logger().error(f"Failed to get transform for {frame_id}: {str(e)}")
            raise

        pose = Pose()
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = t.transform.rotation

        return pose   
    
    def _lock_agv_tray(self, agv_num):
        """
        Lock a tray to the specified AGV.

        This function sends an asynchronous service request to lock a tray
        to the specified AGV, preventing it from moving during transport.

        Args:
            agv_num (int): The AGV number (1-4) to lock the tray on

        Returns:
            bool: True if the service call was initiated successfully,
                  False if the service client was not available
        """
        self.get_logger().info(f"Locking tray to AGV {agv_num}")

        if agv_num not in self._lock_agv_tray_clients:
            self.get_logger().error(f"No lock tray client for AGV {agv_num}")
            return False

        client = self._lock_agv_tray_clients[agv_num]
        request = Trigger.Request()

        future = client.call_async(request)
        future.add_done_callback(
            lambda future: self._lock_agv_tray_callback(future, agv_num)
        )

        # Wait a moment for the lock to take effect
        time.sleep(0.5)
        return True
    
    def _lock_agv_tray_callback(self, future, agv_num):
        """
        Callback for the lock tray service response.

        This function processes the result of the asynchronous lock tray
        service call, logging success or failure messages.

        Args:
            future (rclpy.task.Future): The Future object containing the service response
            agv_num (int): The AGV number the lock operation was performed on
        """
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f"Successfully locked tray to AGV {agv_num}")
            else:
                self.get_logger().warn(f"Failed to lock tray to AGV {agv_num}")
        except Exception as e:
            self.get_logger().error(f"Error calling lock tray service: {str(e)}")
            
            
    def _move_floor_robot_to_pose(self, pose: Pose):
        """
        Move the floor robot to a target pose in Cartesian space.

        This function plans and executes a motion to move the robot's end effector
        to the specified pose. It includes retry logic to handle planning failures.

        Args:
            pose (Pose): Target pose for the robot's end effector

        Returns:
            bool: True if the motion succeeded, False otherwise
        """
        with self._floor_robot_monitor.read_write() as scene:
            self._floor_robot_planner.set_start_state(robot_state=scene.current_state)

            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "world"
            pose_goal.pose = pose
            self._floor_robot_planner.set_goal_state(
                pose_stamped_msg=pose_goal, pose_link="floor_gripper"
            )

        # Limit retries to avoid infinite loops
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            success = self._plan_and_execute(
                self._floor_robot, self._floor_robot_planner, self.get_logger(), "floor_robot"
            )
            if success:
                return True
            attempts += 1
            self.get_logger().warn(
                f"Plan and execute failed, attempt {attempts}/{max_attempts}"
            )
            # Short pause before retry
            time.sleep(0.5)

        self.get_logger().error(f"Failed to move to pose after {max_attempts} attempts")
        return False
    
    
    def _plan_and_execute(
        self,
        robot,
        planning_component,
        logger,
        robot_type,
        single_plan_parameters=None,
        multi_plan_parameters=None,
        sleep_time=0.0,
    ):
        """
        Plan and execute a motion with comprehensive error handling and performance monitoring.

        This helper function handles the complete process of motion planning and execution,
        with detailed logging and timing for performance analysis. It supports both
        single and multi-plan parameter modes for flexible motion specification.

        Args:
            robot: The MoveItPy robot object to execute on
            planning_component: The planning component to use for planning
            logger: Logger object for outputting status messages
            robot_type (str): Type of robot (e.g., "floor_robot")
            single_plan_parameters (dict, optional): Parameters for single plan mode
            multi_plan_parameters (dict, optional): Parameters for multi-plan mode
            sleep_time (float, optional): Time to sleep after execution. Defaults to 0.0.

        Returns:
            bool: True if planning and execution succeeded, False otherwise
        """
        # plan to goal
        logger.debug("Planning trajectory")  # Change to debug level

        # Timing for performance monitoring
        plan_start = time.time()

        # Plan the motion - instead of trying to set parameters directly,
        # we'll use the existing planning functionality with default parameters
        if multi_plan_parameters is not None:
            plan_result = planning_component.plan(
                multi_plan_parameters=multi_plan_parameters
            )
        elif single_plan_parameters is not None:
            plan_result = planning_component.plan(
                single_plan_parameters=single_plan_parameters
            )
        else:
            # Use default planning with no special parameters
            plan_result = planning_component.plan()

        plan_time = time.time() - plan_start
        logger.debug(f"Planning took {plan_time:.3f} seconds")

        # execute the plan
        if plan_result:
            logger.debug("Executing plan")  # Change to debug level
            exec_start = time.time()

            with self._floor_robot_monitor.read_write() as scene:
                scene.current_state.update(True)
                self._floor_robot_state = scene.current_state
                robot_trajectory = plan_result.trajectory

            # Execute with appropriate controllers
            robot.execute(
                robot_trajectory,
                controllers=["floor_robot_controller", "linear_rail_controller"],
            )

            exec_time = time.time() - exec_start
            logger.debug(f"Execution took {exec_time:.3f} seconds")

            # Skip unnecessary sleep
            if sleep_time > 0:
                time.sleep(sleep_time)

            return True
        else:
            logger.error("Planning failed")
            return False
        
    def _move_floor_robot_to_joint_position(self, position_name: str):
        """
        Move the floor robot to a predefined joint position.

        This function plans and executes a motion to move the robot to a named
        joint position configuration, either from predefined positions or
        a special "home" position.

        Args:
            position_name (str): Name of the predefined position or "home"

        Returns:
            bool: True if the motion succeeded, False otherwise
        """
        self.get_logger().info(f"Moving to position: {position_name}")

        try:
            with self._floor_robot_monitor.read_write() as scene:
                # Set the start state
                self._floor_robot_planner.set_start_state(robot_state=scene.current_state)

                # Handle different position types
                if position_name == "home":
                    # For home, we use predefined values
                    home_values = {
                        "linear_actuator_joint": 0.0,
                        "floor_shoulder_pan_joint": 0.0,
                        "floor_shoulder_lift_joint": -1.57,
                        "floor_elbow_joint": 1.57,
                        "floor_wrist_1_joint": -1.57,
                        "floor_wrist_2_joint": -1.57,
                        "floor_wrist_3_joint": 0.0,
                    }

                    # Create a new state for the goal
                    goal_state = copy(scene.current_state)
                    goal_state.joint_positions = home_values

                elif position_name in self._floor_position_dict:
                    # Create a new state for the goal
                    goal_state = copy(scene.current_state)
                    goal_state.joint_positions = self._floor_position_dict[
                        position_name
                    ]

                else:
                    self.get_logger().error(f"Position '{position_name}' not found")
                    return False

                # Create constraint
                joint_constraint = construct_joint_constraint(
                    robot_state=goal_state,
                    joint_model_group=self._floor_robot.get_robot_model().get_joint_model_group(
                        "floor_robot"
                    ),
                )

                # Set goal
                self._floor_robot_planner.set_goal_state(
                    motion_plan_constraints=[joint_constraint]
                )

            # Plan and execute
            success = self._plan_and_execute(
                self._floor_robot, self._floor_robot_planner, self.get_logger(), "floor_robot"
            )

            if success:
                self.get_logger().info(f"Successfully moved to {position_name}")
                return True
            else:
                self.get_logger().error(f"Failed to move to {position_name}")
                return False

        except Exception as e:
            self.get_logger().error(
                f"Error moving to position '{position_name}': {str(e)}"
            )
            return False
        
        
    def _create_floor_joint_position_state(self, joint_positions: list) -> dict:
        """
        Create a dictionary of joint positions for the floor robot.

        This helper function converts a list of joint positions to a dictionary
        mapping joint names to position values for use in motion planning.

        Args:
            joint_positions (list): List of 7 joint positions in order:
                                   [linear_actuator, shoulder_pan, shoulder_lift,
                                    elbow, wrist_1, wrist_2, wrist_3]

        Returns:
            dict: Dictionary mapping joint names to position values
        """
        return {
            "linear_actuator_joint": joint_positions[0],
            "floor_shoulder_pan_joint": joint_positions[1],
            "floor_shoulder_lift_joint": joint_positions[2],
            "floor_elbow_joint": joint_positions[3],
            "floor_wrist_1_joint": joint_positions[4],
            "floor_wrist_2_joint": joint_positions[5],
            "floor_wrist_3_joint": joint_positions[6],
        }
        
    def _attach_model_to_floor_gripper(self, part_to_pick: PartMsg, part_pose: Pose):
        """
        Attach a part model to the floor robot gripper in the planning scene.

        This function creates a collision object for the part and attaches it
        to the robot's gripper in the planning scene, enabling collision-aware
        motion planning with the attached part.

        Args:
            part_to_pick (PartMsg): Part type and color information
            part_pose (Pose): Position and orientation of the part

        Returns:
            bool: True if attachment succeeded, False otherwise
        """
        # Create a part name based on its color and type
        part_name = (
            self._part_colors[part_to_pick.color]
            + "_"
            + self._part_types[part_to_pick.type]
        )

        # Always track the part internally
        self._floor_robot_attached_part = part_to_pick

        # Get the path to the mesh file for the part
        model_path = self._mesh_file_path + self._part_types[part_to_pick.type] + ".stl"

        if not path.exists(model_path):
            self.get_logger().error(f"Mesh file not found: {model_path}")
            return False

        try:
            # Use a single planning scene operation for consistency
            with self._floor_robot_monitor.read_write() as scene:
                # Create the collision object
                co = CollisionObject()
                co.id = part_name
                co.header.frame_id = "world"
                co.header.stamp = self.get_clock().now().to_msg()

                # Create the mesh
                with pyassimp.load(model_path) as assimp_scene:
                    if not assimp_scene.meshes:
                        self.get_logger().error(f"No meshes found in {model_path}")
                        return False

                    mesh = Mesh()
                    # Add triangles
                    for face in assimp_scene.meshes[0].faces:
                        triangle = MeshTriangle()
                        if hasattr(face, "indices"):
                            if len(face.indices) == 3:
                                triangle.vertex_indices = [
                                    face.indices[0],
                                    face.indices[1],
                                    face.indices[2],
                                ]
                                mesh.triangles.append(triangle)
                        else:
                            if len(face) == 3:
                                triangle.vertex_indices = [face[0], face[1], face[2]]
                                mesh.triangles.append(triangle)

                    # Add vertices
                    for vertex in assimp_scene.meshes[0].vertices:
                        point = Point()
                        point.x = float(vertex[0])
                        point.y = float(vertex[1])
                        point.z = float(vertex[2])
                        mesh.vertices.append(point)

                # Add the mesh to the collision object
                co.meshes.append(mesh)
                co.mesh_poses.append(part_pose)
                co.operation = CollisionObject.ADD

                # First add to world - this is important!
                scene.apply_collision_object(co)

                # Then create the attachment
                aco = AttachedCollisionObject()
                aco.link_name = "floor_gripper"
                aco.object = co
                aco.touch_links = [
                    "floor_gripper",
                    "floor_tool0",
                    "floor_wrist_3_link",
                    "floor_wrist_2_link",
                    "floor_wrist_1_link",
                    "floor_flange",
                    "floor_ft_frame",
                ]

                # Update the state
                scene.current_state.attachBody(
                    part_name, "floor_gripper", aco.touch_links
                )
                scene.current_state.update()

                # Make the attachment visible in the planning scene
                ps = PlanningScene()
                ps.is_diff = True
                ps.robot_state.attached_collision_objects.append(aco)

                # Remove from world collision objects since it's now attached
                remove_co = CollisionObject()
                remove_co.id = part_name
                remove_co.operation = CollisionObject.REMOVE
                ps.world.collision_objects.append(remove_co)

                # Apply the complete scene update
                scene.processPlanningSceneMsg(ps)

                self._apply_planning_scene(scene)

            self.get_logger().info(
                f"Successfully attached {part_name} to floor gripper"
            )
            return True

        except Exception as e:
            self.get_logger().error(f"Error attaching model to gripper: {str(e)}")
            return False
        
        
    def _floor_robot_place_part_on_kit_tray(self, agv_num, quadrant):
        """
        Place a part on a kit tray on the specified AGV in the given quadrant.

        This function handles the complete process of:
        1. Verifying a part is currently attached to the gripper
        2. Validating AGV number and quadrant parameters
        3. Moving to the AGV using joint space planning
        4. Positioning the part above the target quadrant using Cartesian planning
        5. Lowering the part into place on the tray
        6. Releasing the part and retreating to a safe position

        Args:
            agv_num (int): AGV number (1-4) to place the part on
            quadrant (int): Quadrant number (1-4) of the tray to place the part in

        Returns:
            bool: True if successful, False otherwise
        """
        if (
            not self._floor_robot_gripper_state
            or not self._floor_robot_gripper_state.attached
        ):
            self.get_logger().error("No part attached")
            return False

        self.get_logger().info(f"Placing part on AGV {agv_num} in quadrant {quadrant}")

        # Validate inputs
        if agv_num < 1 or agv_num > 4:
            self.get_logger().error(f"Invalid AGV number: {agv_num}")
            return False

        if quadrant < 1 or quadrant > 4:
            self.get_logger().error(f"Invalid quadrant number: {quadrant}")
            return False

        # Move to AGV using planning scene monitor
        with self._floor_robot_monitor.read_write() as scene:
            # Set the start state
            self._floor_robot_planner.set_start_state(robot_state=scene.current_state)

            # Create a new state for the goal
            goal_state = copy(scene.current_state)

            # Set joint positions manually
            goal_state.joint_positions = {
                "linear_actuator_joint": self._rail_positions[f"agv{agv_num}"],
                "floor_shoulder_pan_joint": 0.0,
                # Set other joints to reasonable values
                "floor_shoulder_lift_joint": -1.0,
                "floor_elbow_joint": 1.57,
                "floor_wrist_1_joint": -1.57,
                "floor_wrist_2_joint": -1.57,
                "floor_wrist_3_joint": 0.0,
            }

            # Create constraint
            joint_constraint = construct_joint_constraint(
                robot_state=goal_state,
                joint_model_group=self._floor_robot.get_robot_model().get_joint_model_group(
                    "floor_robot"
                ),
            )

            # Set goal
            self._floor_robot_planner.set_goal_state(motion_plan_constraints=[joint_constraint])

        # Plan and execute
        success = self._plan_and_execute(
            self._floor_robot, self._floor_robot_planner, self.get_logger(), "floor_robot"
        )

        if not success:
            self.get_logger().error("Failed to move to AGV")
            return False

        # Continue with placing the part...
        try:
            # Get the AGV tray pose
            agv_tray_pose = self._frame_world_pose(f"agv{agv_num}_tray")

            # Calculate drop position using quadrant offset
            offset_x, offset_y = self._quad_offsets[quadrant]

            # Create cartesian path to place the part
            waypoints = []
            waypoints.append(
                build_pose(
                    agv_tray_pose.position.x + offset_x,
                    agv_tray_pose.position.y + offset_y,
                    agv_tray_pose.position.z + 0.2,  # First move above
                    quaternion_from_euler(0.0, pi, 0.0),
                )
            )

            if not self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, True):
                self.get_logger().error("Failed to move above drop position")
                return False

            # Move down to place the part
            waypoints = []
            waypoints.append(
                build_pose(
                    agv_tray_pose.position.x + offset_x,
                    agv_tray_pose.position.y + offset_y,
                    agv_tray_pose.position.z + 0.15,  # Final placement position
                    quaternion_from_euler(0.0, pi, 0.0),
                )
            )

            if not self._move_floor_robot_cartesian(waypoints, 0.2, 0.2, True):
                self.get_logger().error("Failed to move to place position")
                return False

            # Release part
            self._set_floor_robot_gripper_state(False)
            time.sleep(0.5)  # Wait for release

            # Move up
            waypoints = []
            waypoints.append(
                build_pose(
                    agv_tray_pose.position.x + offset_x,
                    agv_tray_pose.position.y + offset_y,
                    agv_tray_pose.position.z + 0.2,
                    quaternion_from_euler(0.0, pi, 0.0),
                )
            )

            self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, True)

            self.get_logger().info(
                f"Successfully placed part on AGV {agv_num} in quadrant {quadrant}"
            )
            return True

        except Exception as e:
            self.get_logger().error(f"Error placing part: {str(e)}")
            return False


    def _detach_object_from_floor_gripper(self, part_name):
            """
            Detach an object from the floor robot gripper in the planning scene.

            This function removes the attachment between the robot's gripper and
            the specified object in the planning scene, while optionally keeping
            the object in the world model.

            Args:
                part_name (str): Name of the part to detach

            Returns:
                bool: True if detachment succeeded, False otherwise
            """
            self.get_logger().info(f"Detaching {part_name} from floor gripper")

            try:
                with self._floor_robot_monitor.read_write() as scene:
                    # Detach object from robot
                    scene.detachObject(part_name, "floor_gripper")
                    scene.current_state.update()

                    # Optionally remove the object from the world entirely
                    # Uncomment if you want the part to disappear after detachment
                    # collision_object = CollisionObject()
                    # collision_object.id = part_name
                    # collision_object.operation = CollisionObject.REMOVE
                    # scene.apply_collision_object(collision_object)
                    # scene.current_state.update()

                self.get_logger().info(
                    f"Successfully detached {part_name} from floor gripper"
                )
                return True

            except Exception as e:
                self.get_logger().error(f"Error detaching object from gripper: {str(e)}")
            return False