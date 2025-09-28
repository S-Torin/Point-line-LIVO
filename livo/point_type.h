/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file point_type.h
 **/

#pragma once

#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_types.h"

namespace livo {
struct EIGEN_ALIGN16 LiDARPoint {
  PCL_ADD_POINT4D;
  float intensity;
  double time;
  double timebase;
  std::uint32_t is_end;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 LIVOPoint {
  PCL_ADD_POINT4D;
  float intensity;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 HybridPoint {
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  std::uint32_t cnt;
  std::uint32_t colored;
  float score;
  float intensity;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 VisualFeat {
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  PCL_ADD_NORMAL4D;
  float du_x;
  float du_y;
  float du_z;
  float dv_x;
  float dv_y;
  float dv_z;
  float score;
  std::uint8_t is_valid = 0u;
  std::uint8_t is_line = 0u;
  std::uint8_t reserved0 = 0u;
  std::uint8_t reserved = 0u;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 MapVoxel {
  PCL_ADD_POINT4D;
  float vox_size;
  float layer_alpha;
  std::uint32_t vox_layer;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 PosePoint {
  union EIGEN_ALIGN16 {
    float pos[3];
    struct {
      float x, y, z;
    };
  };
  union EIGEN_ALIGN16 {
    float rot[4];
    struct {
      float qx, qy, qz, qw;
    };
  };
  std::uint32_t uid;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace livo

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(livo::LiDARPoint,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (double, time, time)
  (double, timebase, timebase)
  (std::uint32_t, is_end, is_end))

POINT_CLOUD_REGISTER_POINT_STRUCT(livo::LIVOPoint,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity))

POINT_CLOUD_REGISTER_POINT_STRUCT(livo::HybridPoint,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (std::uint32_t, rgba, rgba)
  (std::uint32_t, cnt, cnt)
  (std::uint32_t, colored, colored)
  (float, score, score)
  (float, intensity, intensity))

POINT_CLOUD_REGISTER_POINT_STRUCT(livo::VisualFeat,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (std::uint32_t, rgba, rgba)
  (float, normal_x, normal_x)
  (float, normal_y, normal_y)
  (float, normal_z, normal_z)
  (float, du_x, du_x)
  (float, du_y, du_y)
  (float, du_z, du_z)
  (float, dv_x, dv_x)
  (float, dv_y, dv_y)
  (float, dv_z, dv_z)
  (float, score, score)
  (std::uint8_t, is_valid, is_valid)
  (std::uint8_t, is_line, is_line)
  (std::uint8_t, reserved0, reserved0)
  (std::uint8_t, reserved, reserved))

POINT_CLOUD_REGISTER_POINT_STRUCT(livo::MapVoxel,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, vox_size, vox_size)
  (float, layer_alpha, layer_alpha)
  (std::uint32_t, vox_layer, vox_layer))

POINT_CLOUD_REGISTER_POINT_STRUCT(livo::PosePoint,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, qx, qx)
  (float, qy, qy)
  (float, qz, qz)
  (float, qw, qw)
  (std::uint32_t, uid, uid)
  (double, time, time))
// clang-format on

namespace livo {
using LiDARCloud = pcl::PointCloud<livo::LiDARPoint>;
using LIVOCloud = pcl::PointCloud<livo::LIVOPoint>;
using HybridCloud = pcl::PointCloud<livo::HybridPoint>;
using HybridDeque = std::deque<HybridPoint>;
using VisualCloud = pcl::PointCloud<livo::VisualFeat>;
using VoxelCloud = pcl::PointCloud<livo::MapVoxel>;
using PoseCloud = pcl::PointCloud<livo::PosePoint>;
using ColorPoint = pcl::PointXYZRGB;
using ColorCloud = pcl::PointCloud<pcl::PointXYZRGB>;
using FeatPatch = std::vector<std::vector<float>>;
using ColorPatch = std::vector<std::vector<uint8_t>>;
}  // namespace livo

/*-------- LiDAR types --------*/

namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (float, time, time)
  (std::uint16_t, ring, ring))
// clang-format on

namespace ouster_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  std::uint32_t t;
  std::uint16_t reflectivity;
  std::uint8_t ring;
  std::uint16_t ambient;
  std::uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (std::uint32_t, t, t)
  (std::uint16_t, reflectivity, reflectivity)
  (std::uint8_t, ring, ring)
  (std::uint16_t, ambient, ambient)
  (std::uint32_t, range, range)
)
// clang-format on
