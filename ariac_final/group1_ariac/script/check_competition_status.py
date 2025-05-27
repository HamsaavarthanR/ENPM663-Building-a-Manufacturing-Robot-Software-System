#!/usr/bin/env python3

import rclpy
from group1_ariac.check_competition_status_interface import CompetitionStateCheck
from rclpy.executors import SingleThreadedExecutor

"""
ARIAC Competition State Check

This script initializes an instance of 'CompetitionStateCheck' class registered to a single threaded executor.

If an exception occurs during execution, an error message is logged. The node is 
properly destroyed and ROS 2 is shut down before exiting.

"""

def main(args=None):
    rclpy.init(args=args)
    node = CompetitionStateCheck("status_checker")

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # may want to implement lifecycle check, since node will not longer be needed competition start

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, exiting...\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()