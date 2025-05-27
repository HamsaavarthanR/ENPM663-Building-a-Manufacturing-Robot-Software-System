#!/usr/bin/env python3
import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor
from group1_ariac.bin_detector_interface import SensorsInterface
from time import sleep

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    ## ADD SPECIFIC TASK NODES TO THE EXECUTOR POOL HERE!! ##
    sensors = SensorsInterface('sensors_interface')
    executor.add_node(sensors)
    ## --------------------------------------------------- ##


    # Start spinnig the Multiexecutor Threads Pool
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()


    ## -------- PERFORM SPECIFIC NODE TASKS HERE --------- ##
    # Turns on a debug topic to visualize bounding boxes and slots
    # /ariac/sensors/display_bounding_boxes
    sensors.display_bounding_boxes = True


    sensors.get_logger().info(f"Getting parts from all bins")

    while rclpy.ok():
        try:
            parts_in_bins_log = sensors.log_parts_in_bins()
            for color in parts_in_bins_log.keys():
                for type in parts_in_bins_log[color].keys():
                    sensors.get_logger().info(f"{color} - {type}: ")
                    for pose_info in parts_in_bins_log[color][type]:    
                        sensors.get_logger().info(f"bin_slot: {pose_info['bin_slot']} \n--> part_pose_wrt_world: {pose_info['part_pose_wrt_world']} \n--> part_pose_wrt_camera: {pose_info['part_pose_wrt_camera']}")

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

            sensors.get_logger().info(f"---")

        except KeyboardInterrupt:
            break

        # log every 1 second
        sleep(1.0)
    ## --------------------------------------------------- ##

    sensors.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()