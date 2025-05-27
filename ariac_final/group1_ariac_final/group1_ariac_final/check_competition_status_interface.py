#!/usr/bin/env python3

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ariac_msgs.msg import CompetitionState
from std_srvs.srv import Trigger


class CompetitionStateCheck(Node):
    """
    Class with the service client and subscriber for checking on the competition status

    Attributes:
        _subscriber (Subscriber): ROS2 subscriber object to 'ariac/competition_state' topic.
        _sync_client (Client) : ROS2 Client for 'ariac/start_competition' service.

    Args:
        node_name (str): The name of the node.
    """
    # no need to run in parallel
    mutex_group = MutuallyExclusiveCallbackGroup()
    
    def __init__(self, node_name):
        """
        Initialises the subsciber and client to check the statos of the competition 
        and trigger start request respectively.

        Args:
            node_name (str): The name of the node, passed to the parent Node class.
        """
        super().__init__(node_name)
        # create subscriber to ariac/competition_state/ topic
        self._subscriber = self.create_subscription(
            CompetitionState,           # msg type
            "ariac/competition_state", # topic
            self.competition_state_cb, # callback
            100,                        # QoS
        )

        # create service client for the ariac/start_competition/ service
        self._sync_client = self.create_client(
            Trigger,                # service type
            "ariac/start_competition",    # service name
            callback_group = CompetitionStateCheck.mutex_group  # callback
        )

        # wait until the service exists before trying to check it
        while not self._sync_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("Waiting for competition status service to start...")
        
        # create request
        self._request = Trigger.Request()
        self._logger.info("Server client for competition status created") # can remove this eventually

    def competition_state_cb(self, msg: CompetitionState):
        """
        Callback for subscriber, called when message is received from the subscriber
        """

        # react to the message recieved from the topic
        if msg.competition_state == CompetitionState.READY:
            self.get_logger().info("System is in READY state, invoking service client to start competition...")
            self.send_sync_request()
    
    def send_sync_request(self) -> None:
        """
        Sends synchronous request to the start_competition service
        """
        # wait until the service exists before trying to send a request
        while not self._sync_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for competition status service to start...")
        
        self._sync_client.call(self._request)
        self.get_logger().info("Sent request to start")