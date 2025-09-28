project(run_livo)

add_executable(run_livo run_livo.cc)

target_link_libraries(run_livo
	${catkin_LIBRARIES}
	${livo_LIBRARIES}
	yaml-cpp
	gflags
	glog
	)
