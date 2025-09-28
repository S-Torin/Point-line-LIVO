project(livo_lidar)

add_library(${PROJECT_NAME}
	lidar_handler.cc
	)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
	yaml-cpp
	${PCL_LIBRARIES}
	${catkin_LIBRARIES}
	)
