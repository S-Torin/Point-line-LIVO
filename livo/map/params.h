/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file params.h
 **/

#pragma once

#include <iostream>
#include <vector>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

namespace livo {

struct VoxelParams {
  int32_t max_layer = 3;
  int32_t max_points_num = 1000;
  float point_resolution = 0.05f;
  float max_voxel_size = 1.f;
  std::vector<int32_t> min_point_nums = {5, 5, 5, 5, 5};
  std::vector<float> planer_threshold = {0.01f, 0.1f, 1.f, 1.f, 1.f};
  int32_t update_points_num = 5;
  float surfel_threshold = 0.01f;
};

struct VisualParams {
  float feat_resolution = 0.1f;
};
}