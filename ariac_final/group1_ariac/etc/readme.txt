To run, launch terminals in following order:

- First, run ariac in one terminal using: ros2 launch ariac_gazebo ariac.launch.py trial_name:=rwa2_spring2025
- Then open another terminal and run: ros2 run group1_ariac retrieve_orders.py
- Finally, in another terminal run: ros2 run group1_ariac check_competition_status.py