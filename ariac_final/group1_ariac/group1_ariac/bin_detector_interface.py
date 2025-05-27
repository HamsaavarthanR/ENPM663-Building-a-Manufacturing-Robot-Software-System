#!/usr/bin/env python3


from ament_index_python import get_package_share_directory

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from transformations import translation_matrix
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge, CvBridgeError
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
from ament_index_python import get_package_share_directory
from os import path

from ariac_msgs.msg import (
    CompetitionState as CompetitionStateMsg,
    Part as PartMsg,
    PartPose as PartPoseMsg,
    Order as OrderMsg,
)

from time import sleep

from group1_ariac_msgs.msg import BinPartsPose, BinPartsPoseLot
from ariac_msgs.msg import PartPose
from geometry_msgs.msg import Pose


class SensorsInterface(Node):
    def __init__(self, node_name='sensors_interface'):
        super().__init__(node_name)


        # ROS2 Callback Groups
        self.sensors_cb_group = ReentrantCallbackGroup()

        # Left bins RGB Camera Pose w.r.t world frame --> [(x, y, z), (roll, pitch, yaw)]
        self._left_bins_camera_pose = [(-2.275, -3.0, 1.8),(0.0, np.pi/2, np.pi)]
        # Right bins RGB Camera Pose w.r.t world frame --> [(x, y, z), (roll, pitch, yaw)]
        self._right_bins_camera_pose = [(-2.275, 3.0, 1.8),(0.0, np.pi/2, np.pi)]

        
        # Empty template images for parts to read from 'resources\'
        self.sensor_template = np.ndarray([])
        self.battery_template = np.ndarray([])
        self.pump_template = np.ndarray([])
        self.regulator_template = np.ndarray([])
        
        # HSV colour bounds
        self.HSVcolors = {
            "red"    : {"hmin":   0, "smin":  10, "vmin": 115, "hmax":   4, "smax": 255, "vmax": 255},
            "green"  : {"hmin":  57, "smin":   0, "vmin":   0, "hmax":  80, "smax": 255, "vmax": 255},
            "blue"   : {"hmin": 116, "smin":   0, "vmin": 134, "hmax": 121, "smax": 255, "vmax": 255},
            "orange" : {"hmin":  14, "smin":   0, "vmin": 200, "hmax":  21, "smax": 255, "vmax": 255},
            "purple" : {"hmin": 130, "smin": 180, "vmin": 160, "hmax": 150, "smax": 255, "vmax": 255}
        }

        # BGR reference colours
        self.colors = {
            "red"    : (  0,   0, 255),
            "green"  : (  0, 255,   0),
            "blue"   : (255,   0,   0),
            "orange" : (100, 100, 255),
            "purple" : (255,   0, 100)
        }

        # Part Pose Reporting Object
        self.part_poses = {
            "red"    : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "green"  : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "blue"   : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "orange" : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "purple" : {"battery": [], "pump": [], "sensor": [], "regulator": []}
        }
        # Center of Part Poses
        self.centered_part_poses = {
            "red"    : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "green"  : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "blue"   : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "orange" : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "purple" : {"battery": [], "pump": [], "sensor": [], "regulator": []}
        }

        self.slot_mapping = {
            (1, 1): 1,
            (1, 2): 2,
            (1, 3): 3,
            (2, 1): 4,
            (2, 2): 5,
            (2, 3): 6,
            (3, 1): 7,
            (3, 2): 8,
            (3, 3): 9,
        }

        self.color_mapping = {
            "red"    : 0,
            "green"  : 1,
            "blue"   : 2,
            "orange" : 3,
            "purple" : 4
        }

        self.type_mapping = {
            "battery"   : 10,
            "pump"      : 11,
            "sensor"    : 12,
            "regulator" : 13
        }


        # Turn on debug image publishing for part detection
        self.display_bounding_boxes =  False

        # cv_bridge interface
        self._bridge = CvBridge()

        # Store RGB images from the right bins camera so they can be used for part detection
        self._right_bins_camera_image = None
        self._left_bins_camera_image = None

        # Parts found in the bins
        self._left_bins_parts = []
        self._right_bins_parts = []

        # Read in part templates from 'resources/' folder
        self.sensor_template = cv2.imread(path.join(get_package_share_directory("group1_ariac"), "resources", "sensor.png"), cv2.IMREAD_GRAYSCALE)
        self.regulator_template = cv2.imread(path.join(get_package_share_directory("group1_ariac"), "resources", "regulator.png"), cv2.IMREAD_GRAYSCALE)
        self.battery_template = cv2.imread(path.join(get_package_share_directory("group1_ariac"), "resources", "battery.png"), cv2.IMREAD_GRAYSCALE)
        self.pump_template = cv2.imread(path.join(get_package_share_directory("group1_ariac"), "resources", "pump.png"), cv2.IMREAD_GRAYSCALE)


        # Part-Bins RGBD camera subscribers: (RGB Image Subscriber + Depth Image Subscriber)
        # Right Bins RGB Camera Sub
        self.right_bins_RGB_camera_sub = self.create_subscription(ImageMsg,
                                                                  "/ariac/sensors/right_bins_RGB_camera/rgb_image", #'/ariac/sensors/{sensor_name}/rgb_image'
                                                                  self._right_bins_RGB_camera_cb,
                                                                  qos_profile_sensor_data,
                                                                  callback_group=self.sensors_cb_group
                                                                  )
        # Left Bins RGB Camera Sub
        self.left_bins_RGB_camera_sub = self.create_subscription(ImageMsg,
                                                                  "/ariac/sensors/left_bins_RGB_camera/rgb_image",#'/ariac/sensors/{sensor_name}/rgb_image'
                                                                  self._left_bins_RGB_camera_cb,
                                                                  qos_profile_sensor_data,
                                                                  callback_group=self.sensors_cb_group
                                                                  )
        
        # debug image viewer topic for 'get_bin_parts' function
        self.display_bounding_boxes_pub = self.create_publisher(ImageMsg,
                                              "/ariac/sensors/display_bounding_boxes",
                                              qos_profile_sensor_data,
                                              callback_group=self.sensors_cb_group)
        
        # Create 'PartsBinPoses' Object to store and retrieve pose information of the slots in each bins
        self.partsbinsposes = PartsBinsPoses()
        self._parts_bins_poses = self.partsbinsposes._initialise_parts_in_bins_poses()
        self._parts_bins_poses_wrt_camera = None
        
        # publish poses of parts in bins every second
        self._part_bin_pose_publisher = self.create_publisher(BinPartsPoseLot, "group1_ariac/bin_part_poses",10)
        
        self._part_bin_pose_timer = self.create_timer(1, self.log_parts_in_bins)


    # Using cv_bridge, convert ROS2 ImageMsg types to OpenCV Image representation
    def _left_bins_RGB_camera_cb(self, msg: ImageMsg):
        try:
            self._left_bins_camera_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    def _right_bins_RGB_camera_cb(self, msg: ImageMsg):
        try:
            self._right_bins_camera_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)


    # Log each parts in bins in the '_parts_in_bins' database
    def log_parts_in_bins(self) -> dict:
        
        # message to store bin parts and pose
        msg = BinPartsPoseLot()
        
        bin_parts = None
        # Create a Database to store each parts' pose in the bins
        # "red"    : {"battery": [(bin, slot),(bin,slot),..], 
        #             "pump": [(bin, slot),(bin,slot),..], 
        #             "sensor": [(bin, slot),(bin,slot),..],
        #             "regulator": [(bin, slot),(bin,slot),..]},
        parts_in_bins = {
            "red"    : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "green"  : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "blue"   : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "orange" : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "purple" : {"battery": [], "pump": [], "sensor": [], "regulator": []}
        }

        # Obtain details of parts in each bin and update log
        for bin_number in range(1, 9):
            bin_parts = self.get_bin_parts(bin_number)
            
            # store parts and poses in current bin
            bin_parts_msg = BinPartsPose()

            bin_parts_msg.bin_number = bin_number

            # bin parts will be None unitl image processing starts
            if bin_parts is None:
                self.get_logger().info(f"Waiting for camera imagers ...")
                sleep(1)
            else:
            
                for slot_number, part in bin_parts.items():
                    if part.type is not None:
                        
                        # Part pose message to store pose of part
                        part_pose_msg = PartPose()
                        
                        part_pose_msg.part.color = self.color_mapping[part.color]
                        part_pose_msg.part.type = self.type_mapping[part.type]
                        
                        bin_slot_tup = (bin_number, slot_number)
                        slot = "slot"+str(slot_number)
                        parts_bins_pose_wrt_world = self._parts_bins_poses[bin_number-1][slot]
                        # With respect to right RGB camera
                        if bin_number < 5:
                            camera_pose_wrt_world = self._right_bins_camera_pose
                        elif bin_number >= 5:
                            camera_pose_wrt_world = self._left_bins_camera_pose
                        
                        # Obtain parts in bins pose wrt camera frame
                        parts_bins_poses_wrt_camera = self.partsbinsposes.parts_in_bins_wrt_camera(parts_bins_pose_wrt_world, camera_pose_wrt_world)
                        
                        # storing pose with respect to camera frame
                        pose_msg = Pose()
                        
                        pose_msg.position.x = parts_bins_poses_wrt_camera[0][0]
                        pose_msg.position.y = parts_bins_poses_wrt_camera[0][1]
                        pose_msg.position.z = parts_bins_poses_wrt_camera[0][2]
                        
                        pose_msg.orientation.x = parts_bins_poses_wrt_camera[1][0]
                        pose_msg.orientation.y = parts_bins_poses_wrt_camera[1][1]
                        pose_msg.orientation.z = parts_bins_poses_wrt_camera[1][2]
                        pose_msg.orientation.w = parts_bins_poses_wrt_camera[1][3]     
                        
                        # store pose into part pose message
                        part_pose_msg.pose = pose_msg        
                        
                        
                        parts_in_bins[part.color][part.type].append({"bin_slot": bin_slot_tup, 
                                                                     "part_pose_wrt_world": parts_bins_pose_wrt_world,
                                                                     "part_pose_wrt_camera": parts_bins_poses_wrt_camera})
                        
                        # add part pose to message
                        bin_parts_msg.part_poses.append(part_pose_msg)
                
            # add bin number and part poses to array if any parts were detected
            if len(bin_parts_msg.part_poses) > 0:
                msg.bins.append(bin_parts_msg)
                
        # publish message
        self._part_bin_pose_publisher.publish(msg)
        return parts_in_bins
                        


    def get_bin_parts(self, bin_number: int):
        # Check if the converted image is of Open CV readable Image 
        if type(self._left_bins_camera_image) == type(np.ndarray([])) and \
        type(self._right_bins_camera_image) == type(np.ndarray([])):
            # Create an Image variable based on left or right camera requirement
            if bin_number > 4:
                cv_img = self._left_bins_camera_image
            else:
                cv_img = self._right_bins_camera_image

            imgH, imgW = cv_img.shape[:2]
            
            # roi based on bin number
            if bin_number == 1 or bin_number == 6:
                # bottom left
                cv_img = cv_img[imgH//2:, (imgW//2)+20:imgW-100]
            if bin_number == 2 or bin_number == 5:
                # bottom right
                cv_img = cv_img[imgH//2:, 100:(imgW//2)-20]
            if bin_number == 3 or bin_number == 8:
                # top left
                cv_img = cv_img[:imgH//2, 100:(imgW//2)-20]
            if bin_number == 4 or bin_number == 7:
                # top right
                cv_img = cv_img[:imgH//2, (imgW//2)+20:imgW-100]

            # find parts by colour in image
            self.find_parts(cv_img)

            # debug image view on /ariac/sensors/display_bounding_boxes
            if self.display_bounding_boxes:            
                ros_img = self._bridge.cv2_to_imgmsg(cv_img, "bgr8")
                self.display_bounding_boxes_pub.publish(ros_img)

            # print results
            return self.output_by_slot()

        else:
            # Images are not received, return None
            return None

    def find_parts(self, img):
        '''
        image processing
        ''' 
        # hsv masking
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for color in self.part_poses.keys():
            for type in self.part_poses[color].keys():

                # colour filtering
                imgMask = cv2.inRange(imgHSV, 
                                    self.colorBound(color, "lower"), 
                                    self.colorBound(color, "upper"))

                # template matching
                self.matchTemplate(imgMask, color, type)

                # display bounding boxes
                if self.display_bounding_boxes:
                    if len(self.part_poses[color][type]):
                        # sx, sy -> top left corner
                        # ex, ey -> bottom right corner
                        for (sx, sy, ex, ey) in self.part_poses[color][type]:
                            cv2.rectangle(img, (sx, sy), (ex, ey), self.colors[color], 3)
    
    def matchTemplate(self, imgMask, color, type):
        # template matching
        if type == "pump":
            tH, tW = self.pump_template.shape#[:2]
            matchField = cv2.matchTemplate(imgMask, self.pump_template, cv2.TM_CCOEFF_NORMED)
        elif type == "battery":
            tH, tW = self.battery_template.shape#[:2]
            matchField = cv2.matchTemplate(imgMask, self.battery_template, cv2.TM_CCOEFF_NORMED)
        elif type == "sensor":
            tH, tW = self.sensor_template.shape#[:2]
            matchField = cv2.matchTemplate(imgMask, self.sensor_template, cv2.TM_CCOEFF_NORMED)
        elif type == "regulator":
            tH, tW = self.regulator_template.shape#[:2]
            matchField = cv2.matchTemplate(imgMask, self.regulator_template, cv2.TM_CCOEFF_NORMED)

        # match many
        (yy, xx) = np.where(matchField >= 0.80)

        raw_matches = []
        for (x, y) in zip(xx, yy):
            raw_matches.append((x, y, x+tW, y+tH))

        # non-max suppression
        refined_matches = []
        refined_matches = non_max_suppression(np.array(raw_matches))

        # do this once to save divisions
        htH, htW = tH//2, tW//2
        centered_refined_matches = []
        for sx, sy, _, _ in refined_matches:
            centered_refined_matches.append((sx + htW, sy + htH))

        # store results
        self.part_poses[color][type] = refined_matches
        self.centered_part_poses[color][type] = centered_refined_matches

    def output_by_slot(self):
        bin = dict([(i, PartMsg(color=None, type=None)) for i in range(1, 10)])

        for color in self.centered_part_poses.keys():
            for type in self.centered_part_poses[color].keys():
                for (csx, csy) in self.centered_part_poses[color][type]:
                        row = 0
                        # slot 1, 2, 3
                        if csy <= 88:
                            row = 1                        
                        # slot 7, 8, 9
                        elif csy >= 151: 
                            row = 3
                        # slot 4, 5, 6
                        else: #csy > 88 and csy < 151:
                            row = 2
                        col = 0
                        if csx <= 68:
                            col = 1
                        elif csx >= 131:
                            col = 3
                        else: # csx > 68 and csx < 131:
                            col = 2
                        
                        bin[self.slot_mapping[(row, col)]] = PartMsg(color=color, type=type)
        return bin

    # Helper functions for part detection
    def colorBound(self, color, bound):
        if bound == "lower":
            return np.array([self.HSVcolors[color]["hmin"],
                            self.HSVcolors[color]["smin"],
                            self.HSVcolors[color]["vmin"]])
        # elif bound == "upper":
        return     np.array([self.HSVcolors[color]["hmax"],
                            self.HSVcolors[color]["smax"],
                            self.HSVcolors[color]["vmax"]])

        
            
class PartsBinsPoses():
    def __init__(self):

        self.parts_in_bins_poses = [{"slot1": [], "slot2": [],"slot3": [],"slot4": [],"slot5": [],"slot6": [],"slot7": [],"slot8": [],"slot9": []}, #bin1
                                    {"slot1": [], "slot2": [],"slot3": [],"slot4": [],"slot5": [],"slot6": [],"slot7": [],"slot8": [],"slot9": []}, #bin2
                                    {"slot1": [], "slot2": [],"slot3": [],"slot4": [],"slot5": [],"slot6": [],"slot7": [],"slot8": [],"slot9": []}, #bin3
                                    {"slot1": [], "slot2": [],"slot3": [],"slot4": [],"slot5": [],"slot6": [],"slot7": [],"slot8": [],"slot9": []}, #bin4
                                    {"slot1": [], "slot2": [],"slot3": [],"slot4": [],"slot5": [],"slot6": [],"slot7": [],"slot8": [],"slot9": []}, #bin5
                                    {"slot1": [], "slot2": [],"slot3": [],"slot4": [],"slot5": [],"slot6": [],"slot7": [],"slot8": [],"slot9": []}, #bin6
                                    {"slot1": [], "slot2": [],"slot3": [],"slot4": [],"slot5": [],"slot6": [],"slot7": [],"slot8": [],"slot9": []}, #bin7
                                    {"slot1": [], "slot2": [],"slot3": [],"slot4": [],"slot5": [],"slot6": [],"slot7": [],"slot8": [],"slot9": []} #bin8
                                    ]
        
    def _initialise_parts_in_bins_poses(self):
        # Bin's Center Pose = [(Position Coordinates), (Roll, Pitch, Yaw)] --> with respect to world frame
        bin_centres = [[(-1.9, 3.375, 0.72), (0, 0, np.pi)], #bin1_centre
                      [(-1.9, 2.625, 0.72), (0, 0, np.pi)], #bin2_centre
                      [(-2.65, 2.625, 0.72), (0, 0, np.pi)], #bin3_centre
                      [(-2.65, 3.375, 0.72), (0, 0, np.pi)], #bin4_centre
                      [(-1.9, -3.375, 0.72), (0, 0, np.pi)], #bin5_centre
                      [(-1.9, -2.625, 0.72), (0, 0, np.pi)], #bin6_centre
                      [(-2.65, -2.625, 0.72), (0, 0, np.pi)], #bin7_centre
                      [(-2.65, -3.375, 0.72), (0, 0, np.pi)] #bin8_centre
                      ]

        # Loop over bin1 --> bin8
        for bin in range (0, 8):
            # slot5 --> bin centre (x, y)
            self.parts_in_bins_poses[bin]["slot5"] = bin_centres[bin]
            # slot2 --> x-0.180
            self.parts_in_bins_poses[bin]["slot2"] = [(bin_centres[bin][0][0]-0.180, bin_centres[bin][0][1], bin_centres[bin][0][2]), bin_centres[bin][1]]
            # slot8 --> x+0.180
            self.parts_in_bins_poses[bin]["slot8"] = [(bin_centres[bin][0][0]+0.180, bin_centres[bin][0][1], bin_centres[bin][0][2]), bin_centres[bin][1]]
            # slot4 --> y+0.180
            self.parts_in_bins_poses[bin]["slot4"] = [(bin_centres[bin][0][0], bin_centres[bin][0][1]+0.180, bin_centres[bin][0][2]), bin_centres[bin][1]]
            # slot6 --> y-0.180
            self.parts_in_bins_poses[bin]["slot6"] = [(bin_centres[bin][0][0], bin_centres[bin][0][1]-0.180, bin_centres[bin][0][2]), bin_centres[bin][1]]
            # slot1 --> x-0.180, y+0.180
            self.parts_in_bins_poses[bin]["slot1"] = [(bin_centres[bin][0][0]-0.180, bin_centres[bin][0][1]+0.180, bin_centres[bin][0][2]), bin_centres[bin][1]]
            # slot3 --> x-0.180, y-0.180
            self.parts_in_bins_poses[bin]["slot3"] = [(bin_centres[bin][0][0]-0.180, bin_centres[bin][0][1]-0.180, bin_centres[bin][0][2]), bin_centres[bin][1]]
            # slot7 --> x+0.180, y+0.180
            self.parts_in_bins_poses[bin]["slot7"] = [(bin_centres[bin][0][0]+0.180, bin_centres[bin][0][1]+0.180, bin_centres[bin][0][2]), bin_centres[bin][1]]
            # slot9 --> x+0.180, y-0.180
            self.parts_in_bins_poses[bin]["slot9"] = [(bin_centres[bin][0][0]+0.180, bin_centres[bin][0][1]-0.180, bin_centres[bin][0][2]), bin_centres[bin][1]]

        if self.parts_in_bins_poses is not None:
            return self.parts_in_bins_poses
        
    def rpy_to_rotation_matrix(self, orientation):
        """
        Converts roll, pitch, yaw angles (in radians) to a rotation matrix.

        Args:
            orientation (tuple): (Roll, Pitch, Yaw) angle in radians.

        Returns:
            numpy.ndarray: 3x3 rotation matrix.
        """
        # Obtain Roll, Pitch, Yaw from Orientation
        roll, pitch, yaw = orientation
        # Rotation matrices around the X, Y, and Z axes
        roll_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        pitch_matrix = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        yaw_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        rotation_matrix = np.matmul(yaw_matrix, np.matmul(pitch_matrix, roll_matrix))

        return rotation_matrix
    
    def parts_in_bins_wrt_camera(self, part_pose_wrt_world, camera_pose_wrt_world):
        
        # Create transformation matrix for camera and part w.r.t world frame
        camera_transformation_wrt_world = np.eye(4)
        part_transformation_wrt_world = np.eye(4)

        # camera_pose_wrt_world = [(camera_position_wrt_world), (camera_orientation_wrt_world)]
        camera_position_wrt_world = camera_pose_wrt_world[0] #(x, y, z)
        camera_orientation_wrt_world = camera_pose_wrt_world[1] #(roll, pitch, yaw)

        # Obtian Transformation matrix of the camera frame w.r.t world frame
        # camera_translation_matrix_wrt_world = translation_matrix(camera_position_wrt_world)
        camera_rotation_matrix_wrt_world = self.rpy_to_rotation_matrix(camera_orientation_wrt_world)
        # Combine Translation and Rotation
        # camera_transformation_wrt_world = np.dot(camera_translation_matrix_wrt_world, camera_rotation_matrix_wrt_world)
        camera_transformation_wrt_world[:3,:3] = camera_rotation_matrix_wrt_world
        camera_transformation_wrt_world[:3,3] = np.array(camera_position_wrt_world).reshape(-1,1).flatten()


        # Similarly,
        # parts_pose_wrt_world = [(parts_position_wrt_world), (parts_orientation_wrt_world)]
        part_position_wrt_world = part_pose_wrt_world[0] #(x, y, z)
        part_orientation_wrt_world = part_pose_wrt_world[1] #(roll, pitch, yaw)

        # Obtian Transformation matrix of the part frame w.r.t world frame
        # part_translation_matrix_wrt_world = translation_matrix(part_position_wrt_world)
        part_rotation_matrix_wrt_world = self.rpy_to_rotation_matrix(part_orientation_wrt_world)
        # Combine Translation and Rotation
        # part_transformation_wrt_world = np.dot(part_translation_matrix_wrt_world, part_rotation_matrix_wrt_world)
        part_transformation_wrt_world[:3,:3] = part_rotation_matrix_wrt_world
        part_transformation_wrt_world[:3,3] = np.array(part_position_wrt_world).reshape(-1,1).flatten()



        # Now, obtain transformation of part wrt to camera frame
        # transformation of parts wrt camera = [inverse(transformation of camera wrt world)]*[transformation of parts wrt world]
        try:
            inverse_trans_camera = np.linalg.inv(camera_transformation_wrt_world)
            # Obtain part positoin w.r.t camera as (X, Y, Z)
            part_transformation_wrt_camera = np.dot(inverse_trans_camera, part_transformation_wrt_world)
            part_position_wrt_camera = part_transformation_wrt_camera[:3,3].tolist()
            # Obtain part orientation w.r.t camera as quaternion
            part_rotation_matrix_wrt_camera = part_transformation_wrt_camera[:3,:3]
            part_quaternion_wrt_camera = self.rotation_to_quaternion(part_rotation_matrix_wrt_camera).tolist()

            return (part_position_wrt_camera, part_quaternion_wrt_camera)


        except np.linalg.LinAlgError:
            print("The given matrix is singular anf does not have an inverse!")
            return None
        
    def rotation_to_quaternion(self, rotation_matrix):
        """
        Converts a 3x3 rotation matrix to a quaternion.

        Args:
            rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.

        Returns:
            numpy.ndarray: A quaternion in the format (x, y, z, w).
        """
        # Convert rotation matrix into rotation object
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()

        return quaternion