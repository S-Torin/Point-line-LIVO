project(map)

add_library(${PROJECT_NAME}
	hybrid_map.cc
	hybrid_octree.cc
	)
target_link_libraries(${PROJECT_NAME}
	PUBLIC
	${PCL_LIBRARIES}
	)

set(hybridmap_LIBRARY ${PROJECT_NAME}  CACHE STRING "hybridmap lib" FORCE)
set(hybridmap_LIBRARIES ${PROJECT_NAME}  CACHE STRING "hybridmap lib" FORCE)
