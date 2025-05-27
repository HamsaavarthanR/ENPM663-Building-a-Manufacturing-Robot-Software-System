#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from ariac_msgs.msg import KitTrayPose
from transforms3d import euler
from geometry_msgs.msg import Pose
import PyKDL
import math
from geometry_msgs.msg import (
    Pose,
    PoseStamped, 
    Vector3,
    Quaternion
)

def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    q_msg = Quaternion()
    q_msg.w = q[0]
    q_msg.x = q[1]
    q_msg.y = q[2]
    q_msg.z = q[3]

    return q_msg

class TrayDetector(Node):
    
    def __init__(self):
        super().__init__('tray_detector')
        
        # subscriber to camera info to obtain camera and distortion matrix
        self._left_tray_camera_info = self.create_subscription(CameraInfo, '/ariac/sensors/rgbd_camera_left_tray/camera_info', self.left_tray_info_cb, 10)
        self._right_tray_camera_info = self.create_subscription(CameraInfo, '/ariac/sensors/rgbd_camera_right_tray/camera_info', self.right_tray_info_cb, 10)
        
        # subscriber to tray camera image and depth info
        self._left_tray_camera_image = self.create_subscription(Image, '/ariac/sensors/rgbd_camera_left_tray/rgb_image', self.left_tray_image_cb, 10)
        self._left_tray_camera_depth = self.create_subscription(Image, '/ariac/sensors/rgbd_camera_left_tray/depth_image', self.left_tray_depth_cb, 10)
        self._right_tray_camera_image = self.create_subscription(Image, '/ariac/sensors/rgbd_camera_right_tray/rgb_image', self.right_tray_image_cb, 10)
        self._right_tray_camera_depth = self.create_subscription(Image, '/ariac/sensors/rgbd_camera_right_tray/depth_image', self.left_tray_depth_cb, 10)
        
        # storage for distortion and camera matrices
        self._left_tray_distortion_coeffs = []
        self._right_tray_distortion_coeffs = []
        
        self._left_tray_camera_matrix = []
        self._right_tray_camera_matrix = []
        
        # create a topic and publisher that gives updated part positions
        self._publisher_tray_poses = self.create_publisher(
            KitTrayPose,
            "group1_ariac/tray_part_poses",
            10,
        )

        # # timer for how freqently we publish the updated poses
        # self._tray_pose_update_timer = self.create_timer(
        #     1, # seconds
        #     self.tray_pose_update_timer_cb,
        # )
        
        
        
    def left_tray_image_cb(self, msg:Image):
        """Detect tray ID of trays on left side of conveyor

        Args:
            msg (Image): Image captured by camera
        """
        
        # Convert ROS image to openCV image
        left_tray_cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        
    
        # Convert to grayscale
        left_tray_gray_image = cv.cvtColor(left_tray_cv_image, cv.COLOR_BGR2GRAY)
        
        # detect tray ID using ARUCO markers
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        aruco_params = cv.aruco.DetectorParameters_create()
        corners, ids, _ = cv.aruco.detectMarkers(left_tray_gray_image, aruco_dict,parameters=aruco_params)
        
        if ids is not None:
            cv.aruco.drawDetectedMarkers(left_tray_gray_image, corners, ids)
            self.get_logger().info(f"Detected Tray IDs: {ids.flatten()}")
        else:
            return
        
        # estimate pose using known ARUCO corners
        object_points = np.array([
            [-0.05, -0.05, 0], # Bottom-left corner
            [0.05, -0.05, 0],  # Bottom-right corner
            [0.05, 0.05, 0],  # Top-right corner
            [-0.05, 0.05, 0]  # Top-left corner
        ], dtype=np.float64)
        
            
        camera_quat = euler.euler2quat(np.pi, np.pi/2, 0)
        
        # left camera tray pose
        left = Pose()
        left.position.x = -1.27
        left.position.y = -5.67
        left.position.z =  1.8
        
        left.orientation.x = camera_quat[1]
        left.orientation.y = camera_quat[2]
        left.orientation.z = camera_quat[3]
        left.orientation.w = camera_quat[0]
        # left.orientation.x = .707
        # left.orientation.y = 0.0
        # left.orientation.z = .707
        # left.orientation.w = 0.0
        
        # looping through detected tray ids and corners and estimating pose
        for i in range(len(ids)):
            _, rvec, tvec = cv.solvePnP(object_points, corners[i], self._left_tray_camera_matrix.reshape((3,3)), self._left_tray_distortion_coeffs)
            
            temp = Pose()
            temp.position.x = tvec[0][0]
            temp.position.y = tvec[1][0]
            temp.position.z = tvec[2][0]
            
            quat = euler.euler2quat(rvec[0][0], rvec[1][0], rvec[2][0])
            
            temp.orientation.x = quat[1]
            temp.orientation.y = quat[2]
            temp.orientation.z = quat[3]
            temp.orientation.w = quat[0]
            # temp.orientation.x = 0
            # temp.orientation.y = -0.7
            # temp.orientation.z = 0
            # temp.orientation.w = 0.7
            
            world = self.multiply_pose(left, temp)
            
            # create and publish tray pose message. 
            tray_pose = KitTrayPose()
            
            tray_pose.id = int(ids[i][0])   # needs to be converted to python int otherwise we get an error message. 
            
            tray_pose.pose.position.x = world.position.x + 1.5
            tray_pose.pose.position.y = world.position.y + 0.2
            tray_pose.pose.position.z = world.position.z - 1.25
            
            # storing quaternion values of rotation
            tray_pose.pose.orientation.x = world.orientation.x - world.orientation.x
            tray_pose.pose.orientation.y = world.orientation.y - world.orientation.y
            tray_pose.pose.orientation.w = world.orientation.w - world.orientation.w + 0.0007963
            tray_pose.pose.orientation.z = world.orientation.z + 0.5
            
            self._publisher_tray_poses.publish(tray_pose)
            self.get_logger().info(f"Tray {ids[i][0]} Pose: Translation {[world.position.x, world.position.y, world.position.z]}, Rotation{[world.orientation.x, world.orientation.y, world.orientation.z, world.orientation.w, 5]}")
    
    def left_tray_depth_cb(self, msg:Image):
        """Detect pose of trays on left side of conveyor

        Args:
            msg (Image): Depth info captured by camera
        """
        return
    
    def right_tray_image_cb(self, msg:Image):
        """Detect tray ID of trays on right side of conveyor

        Args:
            msg (Image): Image captured by camera
        """
        
        # Convert ROS image to openCV image
        right_tray_cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        
        # Convert to grayscale
        right_tray_gray_image = cv.cvtColor(right_tray_cv_image, cv.COLOR_BGR2GRAY)
        
        # detect tray ID using ARUCO markers
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        aruco_params = cv.aruco.DetectorParameters_create()
        corners, ids, _ = cv.aruco.detectMarkers(right_tray_gray_image, aruco_dict,parameters=aruco_params)
        
        if ids is not None:
            cv.aruco.drawDetectedMarkers(right_tray_gray_image, corners, ids)
            self.get_logger().info(f"Detected Tray IDs: {ids.flatten()}")
        else:
            return
            
        # estimate pose using known ARUCO corners
        object_points = np.array([
            [-0.05, -0.05, 0], # Bottom-left corner
            [0.05, -0.05, 0],  # Bottom-right corner
            [0.05, 0.05, 0],  # Top-right corner
            [-0.05, 0.05, 0]  # Top-left corner
        ], dtype=np.float64)
    
        
        camera_quat = euler.euler2quat(np.pi, np.pi/2, 0)
        
        # left camera tray pose
        right = Pose()
        right.position.x = -1.247
        right.position.y = 5.81
        right.position.z =  1.8
        
        right.orientation.x = camera_quat[1]
        right.orientation.y = camera_quat[2]
        right.orientation.z = camera_quat[3]
        right.orientation.w = camera_quat[0]
        
        # looping through detected tray ids and corners and estimating pose
        for i in range(len(ids)):
            _, rvec, tvec = cv.solvePnP(object_points, corners[i], self._right_tray_camera_matrix.reshape((3,3)), self._right_tray_distortion_coeffs)
            
            temp = Pose()
            temp.position.x = tvec[0][0]
            temp.position.y = tvec[1][0]
            temp.position.z = tvec[2][0]
            
            quat = euler.euler2quat(rvec[0][0], rvec[1][0], rvec[2][0])
            
            temp.orientation.x = quat[1]
            temp.orientation.y = quat[2]
            temp.orientation.z = quat[3]
            temp.orientation.w = quat[0]
            
            world = self.multiply_pose(right, temp)
            
            # create and publish tray pose message. 
            tray_pose = KitTrayPose()
            
            tray_pose.id = int(ids[i][0])   # needs to be converted to python int otherwise we get an error message. 
            
            tray_pose.pose.position.x = world.position.x + 0.6
            tray_pose.pose.position.y = world.position.y - 0.5
            tray_pose.pose.position.z = world.position.z - 1.04
            
            # storing quaternion values of rotation
            tray_pose.pose.orientation.x = world.orientation.x - world.orientation.x
            tray_pose.pose.orientation.y = world.orientation.y - world.orientation.y
            tray_pose.pose.orientation.w = world.orientation.w + 0.475
            tray_pose.pose.orientation.z = world.orientation.z - world.orientation.z
            
            self.get_logger().info(f"Tray {ids[i][0]} Pose: Translation {[world.position.x, world.position.y, world.position.z]}, Rotation{[world.orientation.x, world.orientation.y, world.orientation.z, world.orientation.w]}")
    
    def right_tray_depth_cb(self, msg:Image):
        """Detect pose of trays on right side of conveyor

        Args:
            msg (Image): Depth info captured by camera
        """
        return
    
    def left_tray_info_cb(self, msg:CameraInfo):
        """Retrieve distortion and coeffs values of left tray camera

        Args:
            msg (CameraInfo): Contains camera pose, and calibration info. 
        """
        
        # retrieve distortion and coeffs values of left tray camera
        self._left_tray_distortion_coeffs = np.array(msg.d, dtype=np.float64)
        self._left_tray_camera_matrix = np.array(msg.k, dtype=np.float64)
        
        return
    
    def right_tray_info_cb(self, msg:CameraInfo):
        """Retrieve distortion and coeffs values of right tray camera

        Args:
            msg (CameraInfo): Contains camera pose, and calibration info. 
        """
        
        # retrieve distortion and coeffs values of left tray camera
        self._right_tray_distortion_coeffs = np.array(msg.d, dtype=np.float64)
        self._right_tray_camera_matrix = np.array(msg.k, dtype=np.float64)
        
        return
    
    def multiply_pose(self,p1: Pose, p2: Pose) -> Pose:
        '''
        Use KDL to multiply two poses together.
        Args:
            p1 (Pose): Pose of the first frame
            p2 (Pose): Pose of the second frame
        Returns:
            Pose: Pose of the resulting frame
        '''

        o1 = p1.orientation
        frame1 = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(o1.x, o1.y, o1.z, o1.w),
            PyKDL.Vector(p1.position.x, p1.position.y, p1.position.z))

        o2 = p2.orientation
        frame2 = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(o2.x, o2.y, o2.z, o2.w),
            PyKDL.Vector(p2.position.x, p2.position.y, p2.position.z))

        frame3 = frame1 * frame2

        # return the resulting pose from frame3
        pose = Pose()
        pose.position.x = frame3.p.x()
        pose.position.y = frame3.p.y()
        pose.position.z = frame3.p.z()

        q = frame3.M.GetQuaternion()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose
    
    # def tray_pose_update_timer_cb(self):
        
    #     """Publish poses of trays to /group1_ariac/tray_part_poses
    #     """
    #     tray_storage = PoseArray()
        
        
    #     self._publisher_tray_poses.publish(tray_storage)
    #     self.get_logger().info("Publishing pose of trays")