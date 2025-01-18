cmake_minimum_required(VERSION 3.0.2)
project(hand_tracking)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  cv_bridge
  image_transport
  realsense2_camera
  gazebo_ros
  gazebo_msgs
)

find_package(gazebo REQUIRED)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    sensor_msgs 
    cv_bridge 
    image_transport 
    realsense2_camera
)

## Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS
  scripts/hand_tracking_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation
install(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_executable(bouncing_object_control src/bouncing_object_control.cpp)
target_link_libraries(bouncing_object_control ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})