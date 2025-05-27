#!/usr/bin/env python3

import rclpy
import threading
from group1_ariac.sensors_interface import SensorsInterface
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    node = SensorsInterface('conveyor_sensors_interface')
    executor.add_node(node)

    # Start spinnig the Multiexecutor Threads Pool
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()
    
    while rclpy.ok():
        try:

            parts_in_conveyor_log = node.log_parts_on_conveyor()

            #node.get_logger().info("Conveyor parts:")
            if node.conveyor_parts:
                count = 0
                for part in node.conveyor_parts:
                    node.get_logger().info(f"--> history ({count}): {part[1].color} - {part[1].type}")
                    count += 1
            

            #for color in parts_in_conveyor_log.keys():
            #    for type in parts_in_conveyor_log[color].keys():
            #        # sensors.get_logger().info(f"{color} - {type}: ")
            #        for pose_info in parts_in_conveyor_log[color][type]:    
            #            sensors.get_logger().info(f"CONVEYOR \n--> part_pose_wrt_world: {pose_info['part_pose_wrt_world']} \n--> part_pose_wrt_camera: {pose_info['part_pose_wrt_camera']}")
            # for bin_number in range(1,9):
            #     bin_parts = sensors.get_bin_parts(bin_number)

            #     # bin_parts will be None until image processing starts
            #     if bin_parts is None:
            #         sensors.get_logger().info(f"Waiting for camera images ...")
            #         sleep(2.0)
            #     else:
            #         sensors.get_logger().info(f"Bin number: {bin_number}")
            #         for _slot_number, _part in bin_parts.items():
            #             if _part.type is None:
            #                 sensors.get_logger().info(f"Slot {_slot_number}: Empty")
            #             else:
            #                 sensors.get_logger().info(f"Slot {_slot_number}: {_part.color} {_part.type}")

            node.get_logger().info(f"...")
        
        except KeyboardInterrupt:
            break

        # log every 1 second
        node.get_clock().sleep_for(Duration(seconds=2.0))

    # try:
    #     executor.spin()
    # except KeyboardInterrupt:
    #     node.get_logger().info("KeyboardInterrupt, exiting...\n")
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()