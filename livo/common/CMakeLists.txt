project(common)

add_library(${PROJECT_NAME}
	ikfom.cc
	mtk/lie_algebra.cc
	mtk/s2.cc
	mtk/so3.cc
	mtk/se3.cc
	)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
	${PCL_LIBRARIES}
	yaml-cpp
	)

set(common_LIBRARY ${PROJECT_NAME} CACHE STRING "common lib" FORCE)
set(common_LIBRARIES ${PROJECT_NAME} CACHE STRING "common lib" FORCE)
