/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file hybrid_map.h
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
#include "map/hybrid_octree.h"
#include "point_type.h"
#include "voxel_hash.h"

namespace livo {
class HybridMap {
 public:
  HybridMap(const YAML::Node& config);
  HybridMap(const VoxelParams vox_param = VoxelParams(),
            const VisualParams vis_param = VisualParams(),
            bool show_log = false)
      : vox_param_(std::make_unique<VoxelParams>(vox_param)),
        vis_param_(std::make_unique<VisualParams>(vis_param)),
        show_log_(show_log) {};
  ~HybridMap();

  void InsertPoints(const HybridCloud::ConstPtr& feats);
  void InsertVisFeats(const VoxelVisualPtr& points);
  VoxelSurfelConstPtr GetNearestSurfel(const LIVOPoint& point) const;
  bool GetNearestSurfels(const LIVOCloud::ConstPtr& points,
                         std::vector<VoxelSurfelConstPtr>& surfels) const;
  bool GetVoxelVisFeats(const std::vector<VOXEL_KEY>& keys,
                        std::vector<VoxelVisualPtr>& feats) const;

  bool empty() const { return hybrid_map_.empty(); }
  VOXEL_KEY voxel_indice(const Eigen::Vector3f& pt) const;
  Eigen::Vector3f voxel_center(const VOXEL_KEY& key) const;
  ColorCloud::Ptr FlattenPoints() const;
  VisualCloud::Ptr FlattenVisFeats() const;
  VoxelCloud::Ptr FlattenMapVoxel() const;
  VoxelCloud::Ptr FlattenMapRootVoxel() const;
  std::vector<VoxelSurfelConstPtr> FlattenSurfels() const;
  std::vector<VOXEL_KEY> FlattenKeys() const;
  void lock_map() { mtx_map_.lock(); }
  void unlock_map() { mtx_map_.unlock(); }

 public:
  std::unique_ptr<VoxelParams> vox_param_ = nullptr;
  std::unique_ptr<VisualParams> vis_param_ = nullptr;

 private:
  bool show_log_ = false;
  std::mutex mtx_map_;
  std::unordered_map<VOXEL_KEY, HybridOctreePtr> hybrid_map_;
  std::vector<VOXEL_KEY> update_keys_vec_;
};
}  // namespace livo