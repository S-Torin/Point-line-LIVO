add_subdirectory(common)
add_subdirectory(sensor)
add_subdirectory(map)

project(livo)

add_library(${PROJECT_NAME}
	livo.cc
	pub_sub.cc)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PCL_LIBRARIES}
	yaml-cpp
	${livo_LIBRARY}
	${common_LIBRARIES}
	${hybridmap_LIBRARY})

set(livo_LIBRARY ${PROJECT_NAME} CACHE STRING "livo lib" FORCE)
set(livo_LIBRARIES ${PROJECT_NAME}  CACHE STRING "livo lib" FORCE)
