project(livo)

add_subdirectory(imu)
add_subdirectory(lidar)
add_subdirectory(cam)

set(livo_LIBRARY livo_imu livo_lidar livo_cam CACHE STRING "livo lib" FORCE)
set(livo_LIBRARIES livo_imu livo_lidar livo_cam CACHE STRING "livo lib" FORCE)
