#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup


from control_msgs.msg import JointControllerState
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Quaternion, PoseStamped, Pose, Point
from ariac_msgs.msg import CompetitionState
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject, PlanningScene

from tf_transformations import quaternion_from_euler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ament_index_python import get_package_share_directory

from moveit import MoveItPy
from moveit.core.robot_state import RobotState, robotStateToRobotStateMsg
from moveit_msgs.srv import ApplyPlanningScene

import time
from os import path
import pyassimp
import math

from ariac_msgs.srv import (
    VacuumGripperControl,
    ChangeGripper,
    MoveAGV,
    PerformQualityCheck,
    SubmitOrder,
)

from rclpy.qos import qos_profile_sensor_data

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

from group1_ariac_msgs.msg import BinPartsPoseLot, ConveyorPartsPoseLot
from group1_ariac.retrieve_orders_interface import OrderClass
from ariac_msgs.msg import KitTrayPose, BinParts, ConveyorParts


# NOTE: use `ros2 param get /<controller_name> joints` to get the joint list for a controller topic

"""
Description: node class for controlling the ceiling robot
"""
class CeilingRobotControl(Node):
    def __init__(self, node_name='ceiling_robot_node'):
        super().__init__(node_name)

        # MoveIt
        self._ceiling_robot = MoveItPy(node_name='ceiling_robot_moveit_py')                         # robot described in MoveIt
        self._ceiling_robot_planner = self._ceiling_robot.get_planning_component("ceiling_robot")   # planner component
        self._ceiling_robot_monitor = self._ceiling_robot.get_planning_scene_monitor()              # scene monitor
        self._ceiling_robot_state = RobotState(self._ceiling_robot.get_robot_model())               # current joint state of the ceiling robot
        
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
        
        '''
        # timer for moving the ceiling robot
        self.controller_timer = self.create_timer(
            1.0,    # Hz
            self._control_cb
        )
        '''

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
        
        # callback groups
        self._reentrant_cb_group = ReentrantCallbackGroup()

        '''
        # create the subcriber to the gantry
        self.gantry_sub = self.create_subscription(
            JointControllerState,
            "/gantry_controller/controller_state",
            self.controller_info_cb,
            100,
        )

        # create the subcriber to the ceiling arm
        self.ceiling_robot_sub = self.create_subscription(
            JointControllerState,
            "/ceiling_robot_controller/controller_state",
            self._controller_info_cb,
            100,
        )
        
        # create a publisher to the gantry joints
        # command for referance (velocities must be at 0.0, positions can be changed):
        # ---> ros2 topic pub /gantry_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, joint_names: ['gantry_x_axis_joint', 'gantry_y_axis_joint', 'gantry_rotation_joint'], points: [{positions: [0.0, 0.0, 0.0], velocities: [0.0, 0.0, 0.0], time_from_start: {sec: 1, nanosec: 0}}]}"
        self.gantry_pub = self.create_publisher(
            JointTrajectory,
            "/gantry_controller/joint_trajectory",
            ReliabilityPolicy.RELIABLE,
        )

        # create a publisher to the ceiling arm joints
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/ceiling_robot_controller/joint_trajectory",
            ReliabilityPolicy.RELIABLE,
        )

        # Joint positioning for different locations/tasks
        # joints: ['gantry_x_axis_joint', 'gantry_y_axis_joint', 'gantry_rotation_joint']
        self._gantry_joint_positions = {
            "home": [2.0, 0.001, -90],      # location where the ceiling robot spawns in at start up

            # gantry goes to the center of the bin group, it's rotation determines which bins it can reach
            "left_bin_bottom":  [4.73,    2.977,  -90],    # reaches bins 5 and 6
            "left_bin_top":     [4.73,    2.977,   90],    # reached bins 7 and 8
            "right_bin_bottom": [4.73,   -1.923,  -90],    # reaches bins 1 and 2
            "right_bin_top":    [4.73,   -1.923,   90]     # reached bins 3 and 4
        }

        # joints: ['ceiling_shoulder_pan_joint', 'ceiling_shoulder_lift_joint', 'ceiling_elbow_joint', 'ceiling_wrist_1_joint', 'ceiling_wrist_2_joint', 'ceiling_wrist_3_joint']
        self._arm_joint_positions = {
            "home": [0, -90, 90, 180, -89, 0],   # joint positions at start up

            # bins are 0.75 m apart in x or y
            # values are arm centered at bin centers, hoving above
            "bin_grab_start_left":  [11, -21, 62, 318, 100, -175],     # for grabbing the bin on the left of the robot
            "bin_grab_start_right": [-31, -14, 56, 318, 658, -175]     # for grabbing the bin on the right of the robot
        }

    # callback for the gantry or arm controller to check positions/velocities
    def _controller_info_cb(msg : JointControllerState):
        # can add something more productive later, just printing to see
        print(f"\nDesired position: {msg.desired.positions}")
        print(f"Desired velocity: {msg.desired.velocities}")
        print(f"Actual position: {msg.actual.positions}")
        print(f"Actual velocity: {msg.actual.velocities}")
            '''
            
        # enable or disable gripper
        self._ceiling_gripper_enable = self.create_client(
            VacuumGripperControl, "/ariac/ceiling_robot_enable_gripper"
        )
        
        self._ceiling_robot_gripper_state = None
        
        
        # subscriber to vacuum gripper state
        self._ceiling_robot_gripper_state_sub = self.create_subscription(
            VacuumGripperState,
            "/ariac/ceiling_robot_gripper_state",
            self._ceiling_robot_gripper_state_cb,
            qos_profile_sensor_data,
            callback_group=self._reentrant_cb_group,
        )
        
         # list to store order messages
        self._order_list = []
        
        #subscriber to orders
        self._order_subscriber = self.create_subscription(OrderMsg, "ariac/orders", self.order_storage_cb, 10, callback_group = self._reentrant_cb_group)
        
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

    def order_storage_cb(self, msg:OrderMsg):       
        """convert order to order class and add to list
        """
        self._order_list.append(OrderClass(msg))

    ##########################################################################################################
    # CEILING ROBOT
    ##########################################################################################################

    def _test_ceiling_robot(self):
        
        # if no order has been found do nothing
        while len(self._order_list) == 0:
            pass
        
        # time waster
        for i in range(10000):
            pass
        
        time.sleep(10)
        # test: go to bin8 center
        test_pose = Pose()
        test_pose.position.x = -2.830000
        test_pose.position.y = -3.195000
        test_pose.position.z =  0.720000 + 0.1
        q = quaternion_from_euler(0.0, math.pi, 0.0)
        test_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self._ceiling_robot_goto(test_pose)
        
        #turn ceiling gripper 
        request = VacuumGripperControl.Request()
        request.enable = True
        # Use call_async with a callback
        future = self._ceiling_gripper_enable.call_async(request)
        
        timeout = 15.0
        start_time = time.time()
        retry_count = 0

        # First try waiting a short time for attachment without moving
        time.sleep(0.1)
        if self._ceiling_robot_gripper_state.attached:
            self.get_logger().info("Part attached on first attempt")
            return True

        # while not self._floor_robot_gripper_state.attached and retry_count < max_retries:
        while not self._ceiling_robot_gripper_state.attached:
            # Move down in larger increments for faster operation
            # z_offset = -0.002 * (retry_count + 1)  # Progressive larger movements
            z_offset = -0.001
            
            test_pose.position.z = test_pose.position.z + z_offset

            self._ceiling_robot_goto(test_pose)

            # Check if attached after movement
            time.sleep(0.4)  # Short wait

            retry_count += 1

            if time.time() - start_time >= timeout:
                self.get_logger().error("Unable to pick up part: timeout")

        if not self._ceiling_robot_gripper_state.attached:
            self.get_logger().error("Unable to pick up part: max retries reached")
        
        # Lift up
        test_pose = Pose()
        test_pose.position.x = -2.830000
        test_pose.position.y = -3.195000
        test_pose.position.z =  0.720000 + 0.5
        q = quaternion_from_euler(0.0, math.pi, 0.0)
        test_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self._ceiling_robot_goto(test_pose)        
        
        
        #wait for attach
        #navigate to quadrant
        test_pose = Pose()
        test_pose.position.x = -2.070000 - 0.15
        test_pose.position.y =  1.200001 + 0.15
        test_pose.position.z =  0.760000 + 0.3
        q = quaternion_from_euler(0.0, math.pi, 0.0)
        test_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self._ceiling_robot_goto(test_pose)
        
        #detach object
        request.enable = False
        # Use call_async with a callback
        future = self._ceiling_gripper_enable.call_async(request)
        
            
    
    def _create_ceiling_plan(self):
        self.get_logger().info("Planning ceiling robot trajectory")
        time_start = time.time()
        
        # assume no special parameters for now
        plan = self._ceiling_robot_planner.plan()
        
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
        with self._ceiling_robot_monitor.read_write() as scene:
            scene.current_state.update(True)
            self._ceiling_robot_state = scene.current_state
            trajectory = plan.trajectory

        self._ceiling_robot.execute(
            trajectory,
            controllers=["ceiling_robot_controller", "gantry_controller"]
        )

        self.get_logger().info(f"---> Total execution time: {(time.time() - time_start):.3f} s")

    def _ceiling_robot_goto(self, pose : Pose):

        with self._ceiling_robot_monitor.read_write() as scene:
            self._ceiling_robot_planner.set_start_state(robot_state=scene.current_state)

            # create the pose message
            target_pose = PoseStamped()
            target_pose.header.frame_id = "world"
            target_pose.pose = pose
            self._ceiling_robot_planner.set_goal_state(
                pose_stamped_msg=target_pose,
                pose_link="ceiling_gripper"
            )
        
        # no infinite looping
        while self._count < 3:
            plan = self._create_ceiling_plan()
            if plan is not None:
                self._execute_plan(plan)
                return True
            else:
                self.get_logger().warn(f"---> Failed attempt ({self._count} of {3})")
                self._count += 1
        
        self.get_logger().error("---> Max attempts reached, failed to move ceiling robot")
        return False
    
    def _ceiling_robot_gripper_state_cb(self, msg: VacuumGripperState):
        """
        Callback for the /ariac/floor_robot_gripper_state topic.

        This function processes gripper state updates, storing the current
        gripper state for use in decision making and planning.

        Args:
            msg (VacuumGripperState): Message containing the current gripper state
        """
        self._ceiling_robot_gripper_state = msg

    ##########################################################################################################
    # MOVEIT
    ##########################################################################################################

    # adds objects to the scene if needed and then waits for the competition to be started before starting moveit
    # parts borrowed from lecture 9
    def _scene_cb(self):
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
            self.get_logger().info("*** NOTE: this is where we would start whatever our task is, I'm testing the ceiling robot movement for now, but that should be replaced eventually")
            self.get_logger().info("\n*********")
            self._test_ceiling_robot()
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
            with self._ceiling_robot_monitor.read_write() as scene:
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
            bin_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

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
            assembly_station_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

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
        conveyor_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

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
        kts1_table_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Aggregate success status
        success = success and self._add_model_to_planning_scene(
            "kts1_table", "kit_tray_table.stl", kts1_table_pose
        )

        kts2_table_pose = Pose()
        kts2_table_pose.position.x = -1.3
        kts2_table_pose.position.y = 5.84
        kts2_table_pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, 0.0)
        kts2_table_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

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