#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from ariac_msgs.msg import KitTrayPose, BinParts, ConveyorParts
from transforms3d import euler
from geometry_msgs.msg import PoseArray, Pose
from ariac_msgs.msg import PartPose, Order, Part
from group1_ariac.retrieve_orders_interface import OrderClass
from group1_ariac.sensors_interface import SensorsInterface
from group1_ariac_msgs.msg import BinPartsPoseLot, ConveyorPartsPoseLot

class Locator(Node):
    """Locates parts and trays needed for orders
    """
    
    def __init__(self):
        """ Initiates suscribers to conveyor, tray, and bin parts. Publishes location info every second
        """
        super().__init__('locator')
        
        # we assume a constant conveyor velocity (known)
        # not robust, but sufficient for now
        self.CONVEYOR_VEL_Y = 0.2

        #suscriber to retrieve tray location
        self._tray_location_subscriber = self.create_subscription(KitTrayPose, "group1_ariac/tray_part_poses", self.tray_location_cb, 10)
        
        # #suscriber to retrieve conveyor part location
        self._conveyor_location_subscriber = self.create_subscription(ConveyorPartsPoseLot, "group1_ariac/conveyor_part_poses", self.conveyor_location_cb, 10)
        
        #suscriber to retrieve bin part location
        self._bin_location_subscriber = self.create_subscription(BinPartsPoseLot, "group1_ariac/bin_part_poses", self.bin_location_cb, 10)
        
        # list to store order messages
        self._order_list = []
        
        #subscriber to orders
        self._order_subscriber = self.create_subscription(Order, "ariac/orders", self.order_storage_cb, 10)
        
        # timer to publish updates
        self._order_timer = self.create_timer(1, self.order_timer_cb)
        
        # storage for location of parts and trays
        self._part_storage = {}
        self._tray_storage = {}
        
        
        # map part color and type to string
        self.color_mapping = {
            Part.RED    : "red",
            Part.GREEN  : "green",
            Part.BLUE   : "blue",
            Part.ORANGE: "orange",
            Part.PURPLE : "purple"
        }

        self.type_mapping = {
            Part.BATTERY   : "battery",
            Part.PUMP     : "pump",
            Part.SENSOR    : "sensor",
            Part.REGULATOR: "regulator"
        }
        
        
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
                
    def order_storage_cb(self, msg:Order):       
        """convert order to order class and add to list
        """
        self._order_list.append(OrderClass(msg))

    
    def order_timer_cb(self):
        """Log Part info
        """
        # loops through order and stores them in order class
        for processed_order in self._order_list:

            #processed_order = OrderClass(order)
            
            # logging order
            self.get_logger().info(f"-Order {processed_order._id}:")
            
            # check if tray has already been found
            if self._tray_storage.get(processed_order._tray_id) is None:
                self.get_logger().info(f"   -Tray {processed_order._tray_id}: Unknown position and orientation")
            else:  
                # extract tray info
                x, y, z, a, b, c, d = self._tray_storage.get(processed_order._tray_id)                
                self.get_logger().info(f"   -Tray {processed_order._tray_id}: [{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
                
            self.get_logger().info(f"   -Parts: ")
            
            # part counter to check how many times a part has been found
            part_counter = {}            
            
            # Check if part has already been found
            for part in processed_order._parts:
                if self._part_storage.get(str(part.part.color) + " " + str(part.part.type)) is None:
                    self.get_logger().info(f"       -{self.color_mapping[part.part.color]} {self.type_mapping[part.part.type]}:")
                    self.get_logger().info(f"           -Location: Unknown position and orientation")
                else:
                    # index to keep track of how many parts have been found
                    part_index = 0
                    
                    if part_counter.get(str(part.part.color) + " " + str(part.part.type)) is not None:
                        part_index = part_counter.get(str(part.part.color) + " " + str(part.part.type))
                    else:
                        part_counter[str(part.part.color) + " " + str(part.part.type)] = 0 
                    
                    # if part is not available exit
                    if len(self._part_storage.get(str(part.part.color) + " " + str(part.part.type))) < (part_index + 1):
                        self.get_logger().info(f"       -{self.color_mapping[part.part.color]} {self.type_mapping[part.part.type]}:")
                        self.get_logger().info(f"           -Location: Unknown position and orientation")
                        continue
                    
                    # extract part info              
                    self.get_logger().info(f"       -{self.color_mapping[part.part.color]} {self.type_mapping[part.part.type]}:")
                    self.get_logger().info(f"           -Location: {self._part_storage.get(str(part.part.color) + ' ' + str(part.part.type))[part_index]['location']}")
                    
                    # check if part is from conveyor or bin
                    if self._part_storage.get(str(part.part.color) + " " + str(part.part.type))[part_index]['location'] != "conveyor": 
                        # extract part storage
                        x, y, z, a, b, c, d = self._part_storage.get(str(part.part.color) + " " + str(part.part.type))[part_index]['pose']                    
                        self.get_logger().info(f"           -[{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
                    else:                                            
                        x, y, z, a, b, c, d = self._part_storage.get(str(part.part.color) + " " + str(part.part.type))[part_index]['pose'] 
                        self.get_logger().info(f"           -First detection: [{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
                        
                        # loop through predicted poses of parts on conveyor belt. 
                        
                        for prediction_index, prediction in enumerate(self._part_storage.get(str(part.part.color) + " " + str(part.part.type))[part_index]['predictions']):
                            x, y, z, a, b, c, d = prediction
                            self.get_logger().info(f"           -Prediction [{prediction_index + 1}s]: [{x}, {y}, {z}] [{a}, {b}, {c}, {d}]")
                        
                    # keep track of how many times part has been found
                    part_counter[str(part.part.color) + " " + str(part.part.type)] += 1
             
        