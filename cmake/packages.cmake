list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Glog REQUIRED)
find_package(Eigen REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL 1.10 REQUIRED COMPONENTS)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV 4 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
	std_msgs
	sensor_msgs
	nav_msgs
	geometry_msgs
	visualization_msgs
	message_generation
	roscpp
	rospy
	tf
	cv_bridge
	)

add_message_files(FILES
	CustomPoint.msg
	CustomMsg.msg
	)

generate_messages(
	DEPENDENCIES
	std_msgs
	)

catkin_package(
	CATKIN_DEPENDS std_msgs sensor_msgs nav_msgs geometry_msgs visualization_msgs roscpp rospy message_runtime cv_bridge
	DEPENDS EIGEN PCL
	INCLUDE_DIRS
)

include_directories(
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	${Glog_INCLUDE_DIRS}
	${yaml-cpp_INCLUDE_DIRS}
	livo
)
