/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file hybrid_octree.h
 **/

#pragma once

#include <deque>
#include <mutex>
#include <vector>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <yaml-cpp/yaml.h>

#include "map/params.h"
#include "point_type.h"
#include "voxel_hash.h"

namespace livo {

struct VoxelSurfel {
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  Eigen::Vector3f color = Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = {0., 0., 1.};  // min_eigen_value
  Eigen::Vector3f u_axis = {1., 0., 0.};  // mid_eigen_value
  Eigen::Vector3f v_axis = {0., 1., 0.};  // max_eigen_value
  Eigen::Vector3f sigma = {1., 1., 1.};   // [min, mid, max]
  Eigen::Matrix3f cov = Eigen::Matrix3f::Identity();
  Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();       // [u, v, n]
  Eigen::Quaternionf init_rotation = Eigen::Quaternionf::Identity();  // [u, v, n]
  bool need_update_ = false;
  int32_t point_nums = 0;
  double d = 0;  // plane coeff
};

struct VoxelVisual {
  VisualCloud feats;
  std::vector<FeatPatch> patches;         // N x layer x patch
  std::vector<ColorPatch> color_patches;  // N x 3 x patch
};

struct VoxelPoints {
  ColorCloud points;
  std::vector<int32_t> cnts;
  std::vector<bool> colored;
};

using VoxelSurfelPtr = std::shared_ptr<VoxelSurfel>;
using VoxelVisualPtr = std::shared_ptr<VoxelVisual>;
using VoxelPointsPtr = std::shared_ptr<VoxelPoints>;
using VoxelSurfelConstPtr = std::shared_ptr<const VoxelSurfel>;
using VoxelVisualConstPtr = std::shared_ptr<const VoxelVisual>;
using VoxelPointsConstPtr = std::shared_ptr<const VoxelPoints>;

class HybridOctree {
 public:
  HybridOctree(const Eigen::Vector3f& vox_center, int32_t layer,
               const VoxelParams& vox_param = VoxelParams(),
               const VisualParams& vis_param = VisualParams());
  ~HybridOctree() = default;

  void InsertPoint(const HybridPoint& point);
  void InsertVisFeat(const VisualFeat& point, const FeatPatch& patch,
                     const ColorPatch& color_patch);
  void UpdateVoxel();
  VoxelSurfelConstPtr GetNearestSurfel(const LIVOPoint& points, double& dist) const;
  std::vector<VoxelVisualPtr> GetVoxelVisFeat() const;

  ColorCloud::Ptr FlattenPoints() const;
  VisualCloud::Ptr FlattenVisFeats() const;
  VoxelCloud::Ptr FlattenMapVoxel(bool only_root = false) const;
  std::vector<VoxelSurfelConstPtr> FlattenSurfels() const;

  int32_t points_size() const { return pts_->points.size(); }

 private:
  void SplitVoxel();
  void UpdateSurfel();
  int32_t child_indice(const Eigen::Vector3f& pt) const;
  Eigen::Vector3f child_center(int32_t indice) const;
  VOXEL_KEY grid_indice(const Eigen::Vector3f& pt) const;
  VOXEL_KEY surfel_grid_indice(const Eigen::Vector3f& pt) const;

 private:
  VoxelParams vox_param_;
  VisualParams vis_param_;

  bool is_root_ = false;
  bool is_leaf_ = true;
  bool is_plane_ = false;
  int32_t layer_ = 0;
  int32_t new_points_num_ = 0;
  double vox_size_ = 1.0;
  Eigen::Vector3f vox_center_;

  VoxelSurfelPtr surf_{new VoxelSurfel};
  VoxelVisualPtr vis_{new VoxelVisual};
  VoxelPointsPtr pts_{new VoxelPoints};
  HybridDeque point_buf_;
  std::unordered_map<VOXEL_KEY, int32_t> grid_;
  std::unordered_map<VOXEL_KEY, int32_t> vis_grid_;
  std::unique_ptr<HybridOctree> childs_[8] = {nullptr};
};

using HybridOctreePtr = std::shared_ptr<HybridOctree>;
}  // namespace livo