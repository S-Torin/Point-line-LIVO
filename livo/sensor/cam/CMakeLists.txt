project(livo_cam)

add_library(${PROJECT_NAME}
	edge_drawing/edge_drawing.cpp
	cam_handler.cc
	)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
	yaml-cpp
	${common_LIBRARIES}
	${OpenCV_LIBRARIES}
	${catkin_LIBRARIES}
	)