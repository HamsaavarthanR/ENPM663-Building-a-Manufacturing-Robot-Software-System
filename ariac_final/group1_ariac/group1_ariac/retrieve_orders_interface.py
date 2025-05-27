#!/usr/bin/env python3

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from ariac_msgs.msg import Order, KittingTask, AssemblyTask, CombinedTask
from ariac_msgs.srv import MoveAGV
from std_srvs.srv import Trigger
from ariac_msgs.srv import SubmitOrder
from ariac_msgs.msg import AGVStatus
from ariac_msgs.msg import CompetitionState

class Color:
    """ 
    Class to store color-coding macros required for terminal outputs.
    """
    # copied from lecture
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    RESET = "\033[0m"

class OrderClass():  
    """
    Class for storing orders as objects.

    Args:
        msg (Order): ARIAC 'Order' message type.

    """
    def __init__(self, msg:Order):
        """
    Obtain 'orders' and initialise order attributes to store them as objects.

        Attributes:
            _id (string): ID - field of 'Order' topic
            _type (uint8): Type - field of 'Order' topic
            _priority (bool): Priority - field of 'Order' topic

            _agv_num (uint8): AGV Number - field of 'Kitting' type 'Order' topic
            _tray_id (int8): Tray ID - field of 'Kitting' type 'Order' topic
            _dest (uint8): Destination - field of 'Kitting' type 'Order' topic

            _agv_nums (list): AGV Number - field of 'Assembly' type 'Order' topic

            _station (uint8): Station Number - field of 'Order' topic applicaple to 'Kitting' and 'Combined' type

            _parts (): Kitting Parts - field 'Order' topic applicable to all three types.

            
        Args:
            msg (Order): ARIAC 'Order' message type.
        """
        self._id = msg.id                # string
        self._type = msg.type            # unit8
        self._priority = msg.priority    # bool

        # kitting
        if self._type == Order.KITTING:
            self._agv_num = msg.kitting_task.agv_number    # uint8
            self._tray_id = msg.kitting_task.tray_id       # int8
            self._dest = msg.kitting_task.destination      # uint8
            self._parts = msg.kitting_task.parts           # ariac_msgs/KittingPart[]
        # assembly
        elif self._type == Order.ASSEMBLY:
            self._agv_nums = msg.assembly_task.agv_numbers  # uint8[]
            self._station = msg.assembly_task.station       # uint8
            self._parts = msg.assembly_task.parts           # ariac_msgs/AssemblyPart[]
        # combined
        elif self._type == Order.COMBINED:
            self._station = msg.combined_task.station       # uint8
            self._parts = msg.combined_task.parts           # ariac_msgs/AssemblyPart[]
        else:
            # This state should never be reached
            pass

class OrderManager(Node):
    """
    Class that manages and stores the incomming orders

    Args:
        node_name (str): Name of the node .
    """
    # desirable so we are always on the lookout for orders
    reent_group = ReentrantCallbackGroup()

    def __init__(self, node_name):
        """

        Attributes:
            _subscriber_state (Subscriber): ROS2 subscriber to 'ariac/competition_state' topic to check when orders are done
            _subscriber_order (Subscriber): ROS2 service client for the ariac/submit_order service
            _submit_client (Client): ROS2 service client for the 'ariac/submit_order' service

            _order_list (list): List to maintain orders obtained
            _priority_list (list): List to maintain priority of order

            _agv1_status_sub (Subscriber): ROS2 subscriber to to detect when AGV1 reach the warehouse
            _agv2_status_sub (Subscriber): ROS2 subscriber to to detect when AGV1 reach the warehouse
            _agv3_status_sub (Subscriber): ROS2 subscriber to to detect when AGV1 reach the warehouse

            _agv1_order_id (string): Variable to dynamically store id of order in AGV1
            _agv2_order_id (string): Variable to dynamically store id of order in AGV2
            _agv3_order_id (string): Variable to dynamically store id of order in AGV3

            _agv1_submit_status (bool): Variable to keep track of whether order on AGV1 has been submitted
            _agv2_submit_status (bool): Variable to keep track of whether order on AGV2 has been submitted
            _agv3_submit_status (bool): Variable to keep track of whether order on AGV3 has been submitted
        
            _has_shipped (bool): flag to ensure orders only ship once

            _end_competitions_client (Client): ROS2 service client for the ariac/end_competition/ service

            _submitted_order_counter (int): Variable to keep track of number of orders submitted

        Args:
            node_name (str): Name of the 'Order Manager' passed to the 'Node' parent class.
        """
        super().__init__(node_name)

        self._order_list = []
        self._priority_list = []
        
        # create subscriber to ariac/competition_state/ topic to check when orders are done
        self._subscriber_state = self.create_subscription(
            CompetitionState,           # msg type
            "ariac/competition_state", # topic
            self.ship, # callback
            100,                        # QoS
        )

        # create subscriber for the orders
        self._subscriber_order = self.create_subscription(
            Order,          # msg type
            "ariac/orders", # topic
            self.order_manager_callback,   # callback
            100,            # QoS
        )
        
        # create service client for the ariac/submit_order service
        self._submit_client = self.create_client(
            SubmitOrder,                # service type
            "ariac/submit_order",    # service name
            callback_group = self.reent_group  # callback
        )
        
        # create subscription to to detect when each of the agvs reach the warehouse
        self._agv1_status_sub = self.create_subscription(AGVStatus, "ariac/agv1_status", self.agv1_status_cb, 10, callback_group=self.reent_group)
        self._agv2_status_sub = self.create_subscription(AGVStatus, "ariac/agv2_status", self.agv2_status_cb, 10, callback_group=self.reent_group)
        self._agv3_status_sub = self.create_subscription(AGVStatus, "ariac/agv3_status", self.agv3_status_cb, 10, callback_group=self.reent_group)
        self._agv4_status_sub = self.create_subscription(AGVStatus, "ariac/agv4_status", self.agv4_status_cb, 10, callback_group=self.reent_group)
    
        # storage to dynamically store id of order in agv
        self._agv1_order_id = "unknown"
        self._agv2_order_id = "unknown"
        self._agv3_order_id = "unknown"
        self._agv4_order_id = "unknown"
        
        
        # keep track of whether order on agv has been submitted
        self._agv1_submit_status = False
        self._agv2_submit_status = False
        self._agv3_submit_status = False
        self._agv4_submit_status = False
        
        # flag to ensure orders only ship once
        self._has_shipped = False
        
        # create service client for the ariac/end_competition/ service
        self._end_competitions_client = self.create_client(
            Trigger,                # service type
            "ariac/end_competition",    # service name
            callback_group = self.reent_group # callback
        )
        
        # keep track of number of orders submitted
        self._submitted_order_counter = 0

    def order_manager_callback(self, msg:Order):
        """
        Callback for subscriber, send the order message to be parsed

        Attributes:
            _error_flag (bool): flag to ensure if any error has occured

        Args:
            msg (Order): ARIAC 'Order' message type.
        """

        self.get_logger().info(Color.BLUE + f"Recieved order {msg.id}" + Color.RESET)
        
        if msg.priority == 1: # if high-prority order
            self.get_logger().info(Color.RED + "---> HIGH Priority" + Color.RESET)

        self._error_flag = False
        if msg.type == Order.KITTING:
            self.get_logger().info("---> Kitting task")
        elif msg.type == Order.ASSEMBLY:
            self.get_logger().info("---> Assembly task")
        elif msg.type == Order.COMBINED:
            self.get_logger().info("---> Combined task")
        else:
            self._error_flag = True
            self.get_logger().error(Color.RED + "Invalid order recieved, ignoring" + Color.RESET)
        
        # add high priority message to high priority list and regular orders to regular list
        if not self._error_flag and msg.priority == 1:
            self._priority_list.append(OrderClass(msg))
        elif not self._error_flag:
            self._order_list.append(OrderClass(msg))
            
            
    def ship(self, msg: CompetitionState):
        """ 
        Class that goes through orders list and ship them

        Args:
            msg (CompetitionState): ARIAC message type that denotes the competition's state.
        """
        
        # since orders are already fulfilled, to ensure that order 2 ships before order 1 we have to wait for all orders to be announced.
        # Otherwise order 1 will be at the warehouse and submitted by the time order 2 arrives. 
        if msg.competition_state == CompetitionState.ORDER_ANNOUNCEMENTS_DONE and (not self._has_shipped):
            
            # combine priority list and order list into one list with priorities in front. 
            full_list = self._priority_list + self._order_list
            
            # extract orders from priority list first then normal orders
            for order in full_list:  
                # obtain agv number from order
                _agv_num = order._agv_num
                
                # store id of order in agv
                if _agv_num == 1:
                    self._agv1_order_id = order._id
                elif _agv_num == 2:
                    self._agv2_order_id = order._id
                elif _agv_num == 3:
                    self._agv3_order_id = order._id
                elif _agv_num == 4:
                    self._agv4_order_id = order._id
                
                # creating lock tray client to lock order tray
                lock_tray_client = self.create_client(
                    Trigger,                # service type
                    f"ariac/agv{_agv_num}_lock_tray",    # service name
                    callback_group=self.reent_group,
                )
                        
                # wait until the service exists before trying to check it
                while not lock_tray_client.wait_for_service(timeout_sec=5.0):
                    self.get_logger().info(f"Waiting for agv{_agv_num} lock tray service to start...")
                    
                #create request to activate lock
                lock_request = Trigger.Request()
                
                
                # call lock and move services()    
                lock_future = lock_tray_client.call_async(lock_request)
                
                while not lock_future.done():
                    pass    
                
                #lock_tray_client.call(lock_request)
                self.get_logger().info(f"agv {order._agv_num} trays locked") 
                
                move_agv_client = self.create_client(
                    MoveAGV,                # service type
                    f"ariac/move_agv{_agv_num}",    # service name
                    callback_group=self.reent_group,
                )
                
                # wait until the service exists before trying to check it
                while not move_agv_client.wait_for_service(timeout_sec=5.0):
                    self.get_logger().info(f"Waiting for move agv{_agv_num} service to start...")
                
                
                # create request to activate move services. 
                move_request = MoveAGV.Request()
                # add destination
                move_request.location = order._dest
                
                move_future = move_agv_client.call_async(move_request)
                
                while not move_future.done():
                    pass
                
                print(move_future.result())     
                   
                #move_agv_client.call(move_request)
                self.get_logger().info(f"agv {order._agv_num} moved to {order._dest}")
                
            self._has_shipped = True
    
        
    def agv1_status_cb(self, msg: AGVStatus):
        """Submit order when AGV1 reaches warehouse

        Args:
            msg (AGVStatus): Contains AGV1 location, pose, and velocity
        """
        
        if msg.location == AGVStatus.WAREHOUSE and (not self._agv1_submit_status):
            # call service
            submit_request = SubmitOrder.Request()
            
            # add order id to be completed
            submit_request.order_id  = self._agv1_order_id
            
            # wait until the service exists before trying to send a request
            while not self._submit_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info("Waiting for submit order service to start...")
                
            self._submit_client.call(submit_request)
            self.get_logger().info(f"Agv 1 arrived at warehouse, order {submit_request.order_id} submitted")
            self._agv1_submit_status = True
            
            # check whether all orde rs have been submitted and competition should be ended
            self._submitted_order_counter += 1
            
            if (len(self._priority_list) + len(self._order_list)) == self._submitted_order_counter:
                self.end_competition()
        else:
            pass
        
        
    def agv2_status_cb(self, msg: AGVStatus):
        """Submit order when AGV2 reaches warehouse

        Args:
            msg (AGVStatus): Contains AGV2 location, pose, and velocity
        """
        
        if msg.location == AGVStatus.WAREHOUSE and (not self._agv2_submit_status):
            # call service
            submit_request = SubmitOrder.Request()
            
            # add order id to be completed
            submit_request.order_id  = self._agv2_order_id
            
            # wait until the service exists before trying to send a request
            while not self._submit_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info("Waiting for submit order service to start...")
                
            self._submit_client.call(submit_request)
            self.get_logger().info(f"Agv 2 arrived at warehouse, order {submit_request.order_id} submitted")
            self._agv2_submit_status = True
            
            # check whether all orde rs have been submitted and competition should be ended
            self._submitted_order_counter += 1
            
            if (len(self._priority_list) + len(self._order_list)) == self._submitted_order_counter:
                self.end_competition()
        else:
            pass
        
    def agv3_status_cb(self, msg: AGVStatus):
        """Submit order when AGV3 reaches warehouse

        Args:
            msg (AGVStatus): Contains AGV3 location, pose, and velocity
        """
        
        if msg.location == AGVStatus.WAREHOUSE and (not self._agv3_submit_status):
            # call service
            submit_request = SubmitOrder.Request()
            
            # add order id to be completed
            submit_request.order_id  = self._agv3_order_id
            
            # wait until the service exists before trying to send a request
            while not self._submit_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info("Waiting for submit order service to start...")
                
            self._submit_client.call(submit_request)
            self.get_logger().info(f"Agv 3 arrived at warehouse, order {submit_request.order_id} submitted")
            self._agv3_submit_status = True
            
            # check whether all orde rs have been submitted and competition should be ended
            self._submitted_order_counter += 1
            
            if (len(self._priority_list) + len(self._order_list)) == self._submitted_order_counter:
                self.end_competition()
        else:
            pass
        
    def agv4_status_cb(self, msg: AGVStatus):
        """Submit order when AGV3 reaches warehouse

        Args:
            msg (AGVStatus): Contains AGV3 location, pose, and velocity
        """
        
        if msg.location == AGVStatus.WAREHOUSE and (not self._agv4_submit_status):
            # call service
            submit_request = SubmitOrder.Request()
            
            # add order id to be completed
            submit_request.order_id  = self._agv4_order_id
            
            # wait until the service exists before trying to send a request
            while not self._submit_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info("Waiting for submit order service to start...")
                
            self._submit_client.call(submit_request)
            self.get_logger().info(f"Agv 4 arrived at warehouse, order {submit_request.order_id} submitted")
            self._agv4_submit_status = True
            
            # check whether all orde rs have been submitted and competition should be ended
            self._submitted_order_counter += 1
            
            if (len(self._priority_list) + len(self._order_list)) == self._submitted_order_counter:
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
        