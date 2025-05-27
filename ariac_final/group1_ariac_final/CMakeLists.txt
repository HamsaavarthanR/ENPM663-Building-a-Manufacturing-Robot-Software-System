cmake_minimum_required(VERSION 3.8)
project(group1_ariac_final)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set package dependencies
set(THIS_PACKAGE_INCLUDE_DEPENDS
  rclcpp
  moveit_ros_planning_interface
  ariac_msgs
  shape_msgs
  control_msgs
  diagnostic_msgs
  tf2_kdl
)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()


# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Install extra files
install(DIRECTORY
  config
  launch
  resources
  meshes
  DESTINATION share/${PROJECT_NAME}
)

# Export libraries
ament_export_libraries(
  ${PROJECT_NAME}_lib
)

# Export include directories
ament_export_include_directories(
  include
)

# Export dependencies
ament_export_dependencies(
  ${THIS_PACKAGE_INCLUDE_DEPENDS}
)


# Python
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  script/check_competition_status.py
  #script/retrieve_orders.py
  #script/sensors.py
  #script/tray_detector.py
  #script/bin_detector.py
  #script/locator.py
  script/ceiling_grab.py
  script/floor_grab.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()