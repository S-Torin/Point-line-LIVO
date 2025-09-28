/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file hybrid_octree.cc
 **/

#include "hybrid_octree.h"

#include <cstring>
#include <unordered_set>
#include <Eigen/Dense>
#include "omp.h"

namespace livo {

using namespace Eigen;

/*----------Hybrid Voxel-----------*/

HybridOctree::HybridOctree(const Vector3f& vox_center, int32_t layer,
                           const VoxelParams& vox_param,
                           const VisualParams& vis_param)
    : vox_center_(vox_center),
      layer_(layer),
      vox_size_(vox_param.max_voxel_size / std::pow(2., layer)) {
  vox_param_ = vox_param;
  vis_param_ = vis_param;
  vox_param_.update_points_num = vox_param_.min_point_nums[layer];
  vox_param_.surfel_threshold = vox_param_.planer_threshold[layer];
  for (int32_t i = 0; i < 8; ++i) childs_[i] = nullptr;
  vis_->feats.reserve(100 / std::pow(2., layer));
  vis_->patches.reserve(100 / std::pow(2., layer));
  vis_->color_patches.reserve(100 / std::pow(2., layer));
  pts_->points.reserve(vox_param_.max_points_num / std::pow(2., layer));
  pts_->cnts.reserve(vox_param_.max_points_num / std::pow(2., layer));
  pts_->colored.reserve(vox_param_.max_points_num / std::pow(2., layer));
  surf_->center = vox_center_;
  surf_->sigma = vox_size_ * 0.5 / 3.f * Eigen::Vector3f::Ones();
}

void HybridOctree::InsertPoint(const HybridPoint& point) {
  if (is_leaf_) {
    point_buf_.emplace_back(point);
  } else {
    int32_t child_id = child_indice(point.getVector3fMap());
    if (childs_[child_id] == nullptr) {
      Vector3f voxel_center = child_center(child_id);
      childs_[child_id].reset(new HybridOctree(
          voxel_center, layer_ + 1, vox_param_, vis_param_));
    }
    childs_[child_id]->InsertPoint(point);
  }
}

void HybridOctree::InsertVisFeat(const VisualFeat& new_feat, const FeatPatch& patch,
                                 const ColorPatch& color_patch) {
  if (is_leaf_) {
    if (!is_plane_) return;
    bool insert_new = true;
    for (int32_t i = 0; i < vis_->feats.size(); ++i) {
      const VisualFeat& map_feat = vis_->feats.at(i);
      if (map_feat.is_line != new_feat.is_line) continue;
      Eigen::Vector3f dist = map_feat.getVector3fMap() - new_feat.getVector3fMap();
      if (dist.norm() < vis_param_.feat_resolution) {
        if (new_feat.score > map_feat.score) {
          vis_->feats.at(i) = new_feat;
          vis_->patches.at(i) = patch;
          vis_->color_patches.at(i) = color_patch;
        }
        insert_new = false;
        break;
      }
    }
    if (insert_new) {
      vis_->feats.emplace_back(new_feat);
      vis_->patches.emplace_back(patch);
      vis_->color_patches.emplace_back(color_patch);
    }
  } else {
    int32_t child_id = child_indice(new_feat.getVector3fMap());
    if (childs_[child_id] != nullptr) {
      childs_[child_id]->InsertVisFeat(new_feat, patch, color_patch);
    }
  }
}

void HybridOctree::UpdateVoxel() {
  if (is_leaf_) {
    while (!point_buf_.empty()) {
      VOXEL_KEY grid_hash;
      if (is_plane_)
        grid_hash = surfel_grid_indice(point_buf_.front().getVector3fMap());
      else
        grid_hash = grid_indice(point_buf_.front().getVector3fMap());
      HybridPoint& new_point = point_buf_.front();
      auto iter = grid_.find(grid_hash);
      if (iter == grid_.end()) {
        int32_t index = points_size();
        grid_[grid_hash] = index;
        ColorPoint color_point;
        color_point.getVector3fMap() = new_point.getVector3fMap();
        color_point.rgb = new_point.rgb;
        pts_->points.emplace_back(color_point);
        pts_->cnts.emplace_back(1);
        pts_->colored.emplace_back(bool(new_point.colored));
      } else {
        int32_t old_index = iter->second;
        pts_->cnts[old_index] += 1;
        Eigen::Vector3f map_xyz = pts_->points[old_index].getVector3fMap();
        Eigen::Vector3f new_xyz = new_point.getVector3fMap();
        map_xyz += (new_xyz - map_xyz) / pts_->cnts[old_index];
        pts_->points[old_index].getVector3fMap() = map_xyz;
        if (!pts_->colored[old_index]) {
          pts_->points[old_index].r = new_point.r;
          pts_->points[old_index].g = new_point.g;
          pts_->points[old_index].b = new_point.b;
          pts_->colored[old_index] = bool(new_point.colored);
        } else if (new_point.colored) {
          Eigen::Vector3f map_rgb{float(pts_->points[old_index].r),
                                  float(pts_->points[old_index].g),
                                  float(pts_->points[old_index].b)};
          Eigen::Vector3f new_rgb{float(new_point.r), float(new_point.g), float(new_point.b)};
          map_rgb += (new_rgb - map_rgb) / pts_->cnts[old_index];
          pts_->points[old_index].r = uint8_t(map_rgb[0]);
          pts_->points[old_index].g = uint8_t(map_rgb[1]);
          pts_->points[old_index].b = uint8_t(map_rgb[2]);
        }
        point_buf_.pop_front();
      }
    }
  } else {
    for (int32_t i = 0; i < 8; ++i) {
      if (childs_[i] != nullptr) childs_[i]->UpdateVoxel();
    }
  }
  UpdateSurfel();
  return;
}

void HybridOctree::SplitVoxel() {
  // LiDAR points
  for (int32_t i = 0; i < pts_->points.size(); ++i) {
    Eigen::Vector3f map_point = pts_->points.at(i).getVector3fMap();
    int32_t child_id = child_indice(map_point);
    if (childs_[child_id] == nullptr) {
      Vector3f voxel_center = child_center(child_id);
      childs_[child_id].reset(new HybridOctree(
          voxel_center, layer_ + 1, vox_param_, vis_param_));
    }
    childs_[child_id]->pts_->points.emplace_back(pts_->points.at(i));
    childs_[child_id]->pts_->cnts.emplace_back(pts_->cnts.at(i));
    childs_[child_id]->pts_->colored.emplace_back(pts_->colored.at(i));
  }
  // Visual points
  for (int32_t i = 0; i < vis_->feats.size(); ++i) {
    const VisualFeat& feat = vis_->feats.at(i);
    const FeatPatch& patch = vis_->patches.at(i);
    const ColorPatch& color_patch = vis_->color_patches.at(i);
    int32_t child_id = child_indice(feat.getVector3fMap());
    if (childs_[child_id] == nullptr) {
      Vector3f voxel_center = child_center(child_id);
      childs_[child_id].reset(new HybridOctree(
          voxel_center, layer_ + 1, vox_param_, vis_param_));
    }
    childs_[child_id]->InsertVisFeat(feat, patch, color_patch);
  }

  for (int32_t i = 0; i < 8; ++i) {
    if (childs_[i] != nullptr) childs_[i]->UpdateVoxel();
  }
  surf_ = nullptr;
  vis_ = nullptr;
  pts_ = nullptr;
  is_plane_ = false;
  is_leaf_ = false;
}

void HybridOctree::UpdateSurfel() {
  if (!is_leaf_) return;
  bool init_plane = is_plane_ == false;
  bool update_plane = false;

  int32_t vox_points_num = pts_->points.size();
  if (vox_points_num < vox_param_.update_points_num ||
      surf_->point_nums >= vox_points_num) return;

  for (int32_t i = surf_->point_nums; i < vox_points_num; ++i) {
    surf_->point_nums += 1;
    Eigen::Vector3f new_xyz = pts_->points.at(i).getVector3fMap();
    surf_->center += (new_xyz - surf_->center) / surf_->point_nums;
    Eigen::Vector3f bias = new_xyz - surf_->center;
    surf_->cov += (bias * bias.transpose() - surf_->cov) / surf_->point_nums;
    if (pts_->colored[i]) {
      Eigen::Vector3f new_color{float(pts_->points.at(i).r),
                                float(pts_->points.at(i).g),
                                float(pts_->points.at(i).b)};
      surf_->color += (new_color - surf_->color) / surf_->point_nums;
    }
  }

  SelfAdjointEigenSolver<Matrix3f> eig(surf_->cov);
  Vector3f evals = eig.eigenvalues().real();
  Matrix3f evecs = eig.eigenvectors().real();
  Matrix3f::Index min_index, max_index;
  evals.rowwise().sum().minCoeff(&min_index);
  evals.rowwise().sum().maxCoeff(&max_index);
  int32_t mid_index = 3 - min_index - max_index;

  surf_->normal = evecs.col(min_index).normalized();
  surf_->v_axis = evecs.col(mid_index).normalized();
  surf_->u_axis = evecs.col(max_index).normalized();
  if ((surf_->u_axis.cross(surf_->v_axis)).dot(surf_->normal) < 0) {
    surf_->normal = -surf_->normal;
  }
  surf_->sigma << std::sqrt(evals(min_index)), std::sqrt(evals(mid_index)),
      std::sqrt(evals(max_index));
  Matrix3f mat;
  mat.col(0) = surf_->u_axis;
  mat.col(1) = surf_->v_axis;
  mat.col(2) = surf_->normal;
  surf_->rotation = Quaternionf(mat);
  surf_->rotation.normalize();
  surf_->d = -(surf_->normal(0) * surf_->center(0) +
               surf_->normal(1) * surf_->center(1) +
               surf_->normal(2) * surf_->center(2));
  is_plane_ = true;
  surf_->need_update_ = false;
  if (surf_->sigma[0] > vox_param_.surfel_threshold &&
      layer_ < vox_param_.max_layer - 1) {
    SplitVoxel();
  }

  if (!is_plane_) return;

  init_plane &= is_plane_;
  Eigen::Vector3f curr_rot = surf_->rotation.matrix().col(2);
  Eigen::Vector3f init_rot = surf_->init_rotation.matrix().col(2);
  update_plane = std::fabs(init_rot.dot(curr_rot)) < std::cos(10.f / 180.f * M_PI);

  if (init_plane || update_plane) {
    surf_->init_rotation = surf_->rotation;
    std::unordered_map<VOXEL_KEY, int32_t> grid_tmp;
    VoxelPointsPtr pts_tmp{new VoxelPoints};
    pts_tmp->points.reserve(pts_->points.size());
    pts_tmp->cnts.reserve(pts_->points.size());
    pts_tmp->colored.reserve(pts_->points.size());
    std::vector<int32_t> points_index_tmp(pts_->points.size(), -1);
    int32_t gs_point_nums = 0;
    for (int32_t i = 0; i < pts_->points.size(); ++i) {
      Eigen::Vector3f map_point = pts_->points.at(i).getVector3fMap();
      VOXEL_KEY grid_hash = surfel_grid_indice(map_point);
      auto iter = grid_tmp.find(grid_hash);
      if (iter == grid_tmp.end()) {
        grid_tmp[grid_hash] = pts_tmp->points.size();
        pts_tmp->points.emplace_back(pts_->points.at(i));
        pts_tmp->cnts.emplace_back(pts_->cnts.at(i));
        pts_tmp->colored.emplace_back(pts_->colored.at(i));
      }
    }
    grid_.swap(grid_tmp);
    pts_ = pts_tmp;
  }
}

VoxelSurfelConstPtr HybridOctree::GetNearestSurfel(
    const LIVOPoint& point, double& dist) const {
  if (is_plane_) {
    dist = std::fabs(surf_->normal.dot(point.getVector3fMap()) + surf_->d);
    return surf_;
  } else if (is_leaf_) {
    return nullptr;
  } else {
    double n_dist = std::numeric_limits<double>::max();
    VoxelSurfelConstPtr n_surf = nullptr;
    for (int32_t i = 0; i < 8; ++i) {
      if (childs_[i] == nullptr) continue;
      double child_dist;
      VoxelSurfelConstPtr child_surf =
          childs_[i]->GetNearestSurfel(point, child_dist);
      if (child_dist < n_dist) {
        n_surf = child_surf;
        n_dist = child_dist;
      }
    }
    dist = n_dist;
    return n_surf;
  }
}

std::vector<VoxelVisualPtr> HybridOctree::GetVoxelVisFeat() const {
  std::vector<VoxelVisualPtr> feats;
  if (is_leaf_ && is_plane_) {
    feats.emplace_back(vis_);
  } else if (!is_leaf_) {
    feats.reserve(std::pow(8, vox_param_.max_layer - layer_ - 1));
    for (int32_t i = 0; i < 8; ++i) {
      if (childs_[i] != nullptr) {
        const std::vector<VoxelVisualPtr>& child_feats = childs_[i]->GetVoxelVisFeat();
        feats.insert(feats.end(), child_feats.begin(), child_feats.end());
      }
    }
  }
  return feats;
}

ColorCloud::Ptr HybridOctree::FlattenPoints() const {
  if (is_leaf_) {
    ColorCloud::Ptr merged_points{new ColorCloud};
    *merged_points = pts_->points;
    return merged_points;
  } else {
    ColorCloud::Ptr merged_points{new ColorCloud};
    for (int32_t i = 0; i < 8; ++i) {
      if (childs_[i] != nullptr) {
        *merged_points += *childs_[i]->FlattenPoints();
      }
    }
    return merged_points;
  }
}

VisualCloud::Ptr HybridOctree::FlattenVisFeats() const {
  VisualCloud::Ptr merged_feats{new VisualCloud};
  if (is_leaf_ && is_plane_) {
    *merged_feats = vis_->feats;
  } else if (!is_leaf_) {
    for (int32_t i = 0; i < 8; ++i) {
      if (childs_[i] != nullptr)
        *merged_feats += *childs_[i]->FlattenVisFeats();
    }
  }
  return merged_feats;
}

VoxelCloud::Ptr HybridOctree::FlattenMapVoxel(bool only_root) const {
  VoxelCloud::Ptr merged_voxels{new VoxelCloud};
  if (is_leaf_ || only_root) {
    MapVoxel map_voxel;
    map_voxel.getVector3fMap() = vox_center_.cast<float>();
    map_voxel.vox_size = vox_size_;
    map_voxel.layer_alpha = float(layer_ + 1) / float(2 * vox_param_.max_layer);
    map_voxel.vox_layer = layer_;
    merged_voxels->emplace_back(map_voxel);
  } else {
    for (int32_t i = 0; i < 8; ++i) {
      if (childs_[i] != nullptr)
        *merged_voxels += *childs_[i]->FlattenMapVoxel();
    }
  }
  return merged_voxels;
}

std::vector<VoxelSurfelConstPtr> HybridOctree::FlattenSurfels() const {
  std::vector<VoxelSurfelConstPtr> surfels;
  if (!is_leaf_) {
    for (int32_t i = 0; i < 8; ++i) {
      if (childs_[i] != nullptr) {
        const auto child_surfels = childs_[i]->FlattenSurfels();
        surfels.insert(surfels.end(), child_surfels.begin(), child_surfels.end());
      }
    }
    return surfels;
  } else if (is_plane_) {
    return std::vector<VoxelSurfelConstPtr>{surf_};
  } else {
    return std::vector<VoxelSurfelConstPtr>();
  }
}

int32_t HybridOctree::child_indice(const Vector3f& pt) const {
  int32_t indice[3] = {0, 0, 0};
  if (pt[0] > vox_center_[0]) indice[0] = 1;
  if (pt[1] > vox_center_[1]) indice[1] = 1;
  if (pt[2] > vox_center_[2]) indice[2] = 1;
  return indice[0] * 4 + indice[1] * 2 + indice[2];
}

Vector3f HybridOctree::child_center(int32_t indice) const {
  Vector3f center = vox_center_;
  double bias = vox_size_ / 4.;
  center[2] += indice & 0x1 ? bias : -bias;
  center[1] += indice & 0x2 ? bias : -bias;
  center[0] += indice & 0x4 ? bias : -bias;
  return center;
}

VOXEL_KEY HybridOctree::grid_indice(const Vector3f& pt) const {
  double loc_xyz[3];
  for (int i = 0; i < 3; i++) {
    loc_xyz[i] = pt[i] / vox_param_.point_resolution;
    if (loc_xyz[i] < 0) {
      loc_xyz[i] -= 1.0;
    }
  }
  return VOXEL_KEY(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
}

VOXEL_KEY HybridOctree::surfel_grid_indice(const Eigen::Vector3f& pt) const {
  Eigen::Matrix3f rfw = surf_->init_rotation.matrix().transpose();
  Eigen::Vector3f uv_surf = Eigen::Vector3f::Zero();
  uv_surf.head<2>() = rfw.block<2, 3>(0, 0) * pt;
  double loc_xyz[3];
  for (int i = 0; i < 3; i++) {
    loc_xyz[i] = uv_surf[i] / vox_param_.point_resolution;
    if (loc_xyz[i] < 0) {
      loc_xyz[i] -= 1.0;
    }
  }
  return VOXEL_KEY(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
}

}  // namespace livo