/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file hybrid_map.cc
 **/

#include "hybrid_map.h"

#include <unordered_set>
#include "omp.h"
#include <Eigen/Dense>

namespace livo {

using namespace Eigen;
/*----------Hybrid Map-----------*/

HybridMap::HybridMap(const YAML::Node& config) {
  vox_param_.reset(new VoxelParams);
  vox_param_->max_layer = config["max_layer"].as<int32_t>();
  vox_param_->max_voxel_size = config["max_voxel_size"].as<float>();
  vox_param_->max_points_num = config["max_point_num"].as<int32_t>();
  vox_param_->planer_threshold = config["planer_threshold"].as<std::vector<float>>();
  vox_param_->min_point_nums = config["min_point_nums"].as<std::vector<int32_t>>();
  vox_param_->point_resolution = config["point_resolution"].as<float>();
  show_log_ = config["show_log"].as<bool>(false);
  vis_param_.reset(new VisualParams);
  vis_param_->feat_resolution = config["feat_resolution"].as<float>(0.1);
}

HybridMap::~HybridMap() {
  std::cout << "\033[32mHybridMap Destruction!\033[0m" << std::endl;
}

void HybridMap::InsertPoints(const HybridCloud::ConstPtr& points) {
  double t0 = omp_get_wtime();
  std::unordered_set<VOXEL_KEY> update_keys;
  for (const HybridPoint& point : points->points) {
    VOXEL_KEY voxel_hash = voxel_indice(point.getVector3fMap());
    auto iter = hybrid_map_.find(voxel_hash);
    if (iter == hybrid_map_.end()) {
      Vector3f vox_center = {
          (float(voxel_hash.x) + 0.5f) * vox_param_->max_voxel_size,
          (float(voxel_hash.y) + 0.5f) * vox_param_->max_voxel_size,
          (float(voxel_hash.z) + 0.5f) * vox_param_->max_voxel_size};
      hybrid_map_[voxel_hash] = std::make_shared<HybridOctree>(
          vox_center, 0, *vox_param_, *vis_param_);
      hybrid_map_[voxel_hash]->InsertPoint(point);
    } else {
      iter->second->InsertPoint(point);
    }
    update_keys.insert(voxel_hash);
  }
  double t1 = omp_get_wtime();
  lock_map();
  update_keys_vec_ = std::vector<VOXEL_KEY>(update_keys.begin(), update_keys.end());
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(none) shared(update_keys_vec_, hybrid_map_)
#endif
  for (auto it = update_keys_vec_.begin(); it != update_keys_vec_.end(); it++) {
    hybrid_map_[*it]->UpdateVoxel();
  }
  double t2 = omp_get_wtime();
  unlock_map();
  if (show_log_) {
    std::cout << "\033[32mInsert Hybrid Map " << points->size() << " Points: "
              << "iterate and insert cost " << (t1 - t0) * 1000. << " ms, "
              << "update voxel cost " << (t2 - t1) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
}

void HybridMap::InsertVisFeats(const VoxelVisualPtr& feats) {
  if (feats == nullptr || feats->feats.size() == 0) return;
  double t0 = omp_get_wtime();
  for (int32_t i = 0; i < feats->feats.points.size(); ++i) {
    const VisualFeat& point = feats->feats.points.at(i);
    const FeatPatch& patch = feats->patches.at(i);
    const ColorPatch& color_patch = feats->color_patches.at(i);
    VOXEL_KEY voxel_hash = voxel_indice(point.getVector3fMap());
    auto iter = hybrid_map_.find(voxel_hash);
    if (iter != hybrid_map_.end()) {
      iter->second->InsertVisFeat(point, patch, color_patch);
    }
  }
  double t1 = omp_get_wtime();
  if (show_log_) {
    std::cout << "\033[32mInsert Hybrid Map " << feats->feats.points.size()
              << " Visual Feats cost " << (t1 - t0) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
}

VoxelSurfelConstPtr HybridMap::GetNearestSurfel(const LIVOPoint& point) const {
  VOXEL_KEY voxel_hash = voxel_indice(point.getVector3fMap());
  Vector3f center = voxel_center(voxel_hash);
  auto iter = hybrid_map_.find(voxel_hash);
  if (iter == hybrid_map_.end()) {
    iter = hybrid_map_.find(voxel_indice(2 * point.getVector3fMap() - center));
  }
  if (iter != hybrid_map_.end()) {
    double dist;
    return iter->second->GetNearestSurfel(point, dist);
  }
  return nullptr;
}

bool HybridMap::GetNearestSurfels(const LIVOCloud::ConstPtr& points,
                                  std::vector<VoxelSurfelConstPtr>& surfels) const {
  double t0 = omp_get_wtime();
  surfels.resize(points->size(), nullptr);
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(none) shared(points, surfels, hybrid_map_)
#endif
  for (int32_t i = 0; i < points->size(); ++i) {
    surfels.at(i) = GetNearestSurfel(points->at(i));
  }
  double t1 = omp_get_wtime();
  if (show_log_) {
    std::cout << "\033[32mGet Nearest Surfels of " << points->size()
              << " Points cost " << (t1 - t0) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
  return true;
}

bool HybridMap::GetVoxelVisFeats(const std::vector<VOXEL_KEY>& keys,
                                 std::vector<VoxelVisualPtr>& feats) const {
  double t0 = omp_get_wtime();
  std::vector<std::vector<VoxelVisualPtr>> feats_vec;
  feats_vec.resize(keys.size());
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(none) shared(hybrid_map_, feats_vec, keys)
#endif
  for (int32_t i = 0; i < keys.size(); ++i) {
    VOXEL_KEY voxel_hash = keys.at(i);
    auto iter = hybrid_map_.find(voxel_hash);
    if (iter != hybrid_map_.end()) {
      feats_vec.at(i) = iter->second->GetVoxelVisFeat();
    }
  }
  int32_t feats_nums = 0;
  for (const auto& vec : feats_vec)
    feats_nums += vec.size();
  feats.reserve(feats_nums);
  for (const auto& vec : feats_vec)
    for (const auto& feat : vec)
      feats.emplace_back(feat);
  double t1 = omp_get_wtime();
  if (show_log_) {
    std::cout << "\033[32mGet " << feats.size() << " Visual Features of "
              << keys.size() << " Voxels cost " << (t1 - t0) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
  return true;
}

VOXEL_KEY HybridMap::voxel_indice(const Vector3f& pt) const {
  float loc_xyz[3];
  for (int i = 0; i < 3; i++) {
    loc_xyz[i] = pt[i] / vox_param_->max_voxel_size;
    if (loc_xyz[i] < 0) {
      loc_xyz[i] -= 1.0;
    }
  }
  return VOXEL_KEY(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
}

Vector3f HybridMap::voxel_center(const VOXEL_KEY& key) const {
  return Vector3f{(float(key.x) + 0.5f) * float(vox_param_->max_voxel_size),
                  (float(key.y) + 0.5f) * float(vox_param_->max_voxel_size),
                  (float(key.z) + 0.5f) * float(vox_param_->max_voxel_size)};
}

ColorCloud::Ptr HybridMap::FlattenPoints() const {
  if (hybrid_map_.empty()) return nullptr;
  double t0 = omp_get_wtime();
  ColorCloud::Ptr merge_cloud{new ColorCloud};
  std::vector<ColorCloud::ConstPtr> voxel_clouds;
  voxel_clouds.reserve(hybrid_map_.size());
  int32_t merge_cloud_size = 0;
  for (auto it = hybrid_map_.begin(); it != hybrid_map_.end(); ++it) {
    voxel_clouds.emplace_back(it->second->FlattenPoints());
    merge_cloud_size += voxel_clouds.back()->size();
  }
  merge_cloud->reserve(merge_cloud_size);
  for (const auto& vox_cloud : voxel_clouds) {
    merge_cloud->insert(merge_cloud->end(), vox_cloud->begin(), vox_cloud->end());
  }
  double t1 = omp_get_wtime();
  if (show_log_) {
    std::cout << "\033[32mFlatten Hybrid Map " << merge_cloud_size << " Points "
              << "cost " << (t1 - t0) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
  return merge_cloud;
}

VisualCloud::Ptr HybridMap::FlattenVisFeats() const {
  if (hybrid_map_.empty()) return nullptr;
  double t0 = omp_get_wtime();
  VisualCloud::Ptr merge_feats{new VisualCloud};
  std::vector<VisualCloud::ConstPtr> voxel_feats;
  voxel_feats.reserve(hybrid_map_.size());
  int32_t merge_feats_size = 0;
  for (auto it = hybrid_map_.begin(); it != hybrid_map_.end(); ++it) {
    voxel_feats.emplace_back(it->second->FlattenVisFeats());
    merge_feats_size += voxel_feats.back()->size();
  }
  merge_feats->reserve(merge_feats_size);
  for (const auto& vox_feat : voxel_feats) {
    merge_feats->insert(merge_feats->end(), vox_feat->begin(), vox_feat->end());
  }
  double t1 = omp_get_wtime();
  if (show_log_) {
    std::cout << "\033[32mFlatten Hybrid Map " << merge_feats_size << " Visual Feats "
              << "cost " << (t1 - t0) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
  return merge_feats;
}

VoxelCloud::Ptr HybridMap::FlattenMapVoxel() const {
  if (hybrid_map_.empty()) return nullptr;
  double t0 = omp_get_wtime();
  VoxelCloud::Ptr merged_voxels{new VoxelCloud};
  int32_t merged_voxels_size = 0;
  for (auto it = hybrid_map_.begin(); it != hybrid_map_.end(); ++it) {
    *merged_voxels += *(it->second->FlattenMapVoxel());
  }
  double t1 = omp_get_wtime();
  if (show_log_) {
    std::cout << "\033[32mFlatten Hybrid Map " << merged_voxels_size << " Voxels "
              << "cost " << (t1 - t0) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
  return merged_voxels;
}

VoxelCloud::Ptr HybridMap::FlattenMapRootVoxel() const {
  if (hybrid_map_.empty()) return nullptr;
  double t0 = omp_get_wtime();
  VoxelCloud::Ptr merged_voxels{new VoxelCloud};
  int32_t merged_voxels_size = 0;
  for (auto it = hybrid_map_.begin(); it != hybrid_map_.end(); ++it) {
    *merged_voxels += *(it->second->FlattenMapVoxel(true));
  }
  double t1 = omp_get_wtime();
  if (show_log_) {
    std::cout << "\033[32mFlatten Hybrid Map " << merged_voxels_size << " Root Voxels "
              << "cost " << (t1 - t0) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
  return merged_voxels;
}

std::vector<VoxelSurfelConstPtr> HybridMap::FlattenSurfels() const {
  if (hybrid_map_.empty()) return std::vector<VoxelSurfelConstPtr>();
  double t0 = omp_get_wtime();
  std::vector<VoxelSurfelConstPtr> merge_surfels;
  for (auto it = hybrid_map_.begin(); it != hybrid_map_.end(); ++it) {
    const auto voxel_surfels = it->second->FlattenSurfels();
    merge_surfels.insert(merge_surfels.end(), voxel_surfels.begin(), voxel_surfels.end());
  }
  double t1 = omp_get_wtime();
  if (show_log_) {
    std::cout << "\033[32mFlatten Hybrid Map " << merge_surfels.size() << " Planes "
              << "cost " << (t1 - t0) * 1000. << " ms."
              << "\033[0m" << std::endl;
  }
  return merge_surfels;
}

std::vector<VOXEL_KEY> HybridMap::FlattenKeys() const {
  if (hybrid_map_.empty()) return std::vector<VOXEL_KEY>();
  std::vector<VOXEL_KEY> keys;
  keys.reserve(hybrid_map_.size());
  for (auto it = hybrid_map_.begin(); it != hybrid_map_.end(); ++it) {
    keys.emplace_back(it->first);
  }
  return keys;
}

}  // namespace livo