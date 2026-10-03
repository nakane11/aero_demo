cmake_minimum_required(VERSION 3.0.2)
project(aero_demo)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

## src/aero_demo/ の ROS 非依存ライブラリを import 可能にする
catkin_python_setup()

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
