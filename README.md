# ENPM663 - Building a Manufacturing Robot Software System
ARIAC - Final Project

![Ariac-Demo](https://github.com/user-attachments/assets/10e12e25-5951-47c0-839b-bc648005b1ce)

## Overview

The Agile Robotics for Industrial Automation Competition (ARIAC) is an annual robotics challenge hosted by the National Institute of Standards and Technology (NIST) since June 2017, designed to push the boundaries of industrial robot agility in a simulated manufacturing setting. Participants develop a Competitor Control System (CCS) that interfaces with an ARIAC Manager (AM) within a containerized Docker environment to execute trials, leveraging ROS 2 Iron on Ubuntu 22.04 for seamless integration. The competition unfolds in a dynamic Gazebo simulation where teams must perform pick-and-place, assembly, and kitting tasks, coordinating multiple robot types—including autonomous guided vehicles (AGVs), floor robots, and ceiling gantry systems—to mirror real-world factory floors. ARIAC incorporates a suite of agility challenges—faulty parts, flipped parts, dropped parts, robot malfunctions, sensor blackouts, high-priority orders, and insufficient parts—to emulate the unpredictability of modern manufacturing. Scoring combines performance, efficiency, and cost metrics into a unified framework, making ARIAC a standardized testbed for benchmarking autonomous robotic algorithms and shaping future standards in manufacturing agility. 

<img width="468" alt="Picture 1" src="https://github.com/user-attachments/assets/5fa51520-316d-4b41-803f-2c5fbd78b914" />

The specific objectives of the final project were to perform a kitting task using both the ceiling and the floor robot, and then ship both the robots to the warehouse, submit the order, score the order, and then end the competition. The kitting task involved picking up parts from the bin using both the floor and the ceiling robot, placing it on their respective trays, and then picking and placing those trays on the AGVs using the floor robot.


## Prerequisites
- Operating System
  * `Ubuntu 22.04 (Jammy Jellyfish)`
- ROS 2 Iron
  * [Iron Irwini](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debs.html)
- ARIAC 2024 Installation
  * [ARIAC 2024.5.0](https://pages.nist.gov/ARIAC_docs/en/latest/getting_started/installation.html)


## Run Final Project Package
Place 'ariac_final' in your '/src' folder in the ARIAC workspace

Dependencies:
- Install the following dependencies before running the executables
  ```bash
  pip install transforms3d
  ```
  ```bash
  pip install transformations
  ```
  
To Build:
- build competitor package using
  ```bash
  colcon build --packages-select group1_ariac_final
  ```
- build custom messages with
  ```bash
  colcon build --packages-select group1_ariac_msgs
  ```
- build competitor package using 
  ```bash
  colcon build --packages-select group1_ariac
  ```
- Source current workspace
  ```bash
  source install/setup.bash
  ```
  
To Launch: 
- to launch Gazebo:
  ```bash
  ros2 launch ariac_gazebo ariac.launch.py trial_name:=final_project competitor_pkg:=group1_ariac_final
  ```
- to launch our code (_please add your node to the launch file!_)
  ```bash
  ros2 launch group1_ariac_final group1_ariac_final.launch.py program:=python rviz:=<true/false>
  ```

To Run Executables:
  ```bash
  ros2 run group1_ariac tray_detector.py
  ```
  ```bash
  ros2 run group1_ariac bin_detector.py
  ```

To Start:
- start the compeition manually (the moveit planner should start automatically after)
  ```bash
  ros2 run group1_ariac_final check_competition_status.py
  ```

## Team

1. Anne-Michelle Lieberson
2. Hamsaavarthan Ravichandar
3. Manas Desai
4. Robens Cyprien

## Links

- ARIAC Documentation — [ARIAC Docs 2024.5.0 documentation. (2024). Nist.gov; NIST.](https://pages.nist.gov/ARIAC_docs/en/latest/index.html)
- Video Demonstration — [YouTube](https://youtu.be/nRMoOkfuviY)

