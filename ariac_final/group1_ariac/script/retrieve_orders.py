#!/usr/bin/env python3

import rclpy
from group1_ariac.retrieve_orders_interface import OrderManager
from rclpy.executors import MultiThreadedExecutor

"""
ARIAC Retrieve Order

This script initializes an instance of 'OrderManager' class registered to a multi threaded executor.

If an exception occurs during execution, an error message is logged. The node is 
properly destroyed and ROS 2 is shut down before exiting.

"""

def main(args=None):
    rclpy.init(args=args)
    node = OrderManager("order_manager")

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, exiting...\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()