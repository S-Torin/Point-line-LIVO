project(livo_imu)

add_library(${PROJECT_NAME}
	imu_handler.cc
	imu_predict_func.cc
	)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
	yaml-cpp
	${common_LIBRARIES}
	${catkin_LIBRARIES}
	)
