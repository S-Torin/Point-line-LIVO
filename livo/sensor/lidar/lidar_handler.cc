/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file lidar_handler.cc
 **/

#include "lidar_handler.h"

#include <cmath>
#include <limits>
#include <vector>

#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <omp.h>

#include "common/mtk/lie_algebra.h"
#include "log.h"
#include "voxel_hash.h"

namespace livo {

namespace {
const double kMathEpsilon = 1.e-12;
}  // namespace

using namespace Eigen;

LidarHandler::LidarHandler(const YAML::Node& config, ros::NodeHandle& nh,
                           std::shared_ptr<HybridMap> local_map)
    : local_map_(local_map) {
  enable_ = config["enable"].as<bool>(true);
  pcd_save_en_ = config["pcd_save_en"].as<bool>(true);
  lidar_type_ = config["lidar_type"].as<std::string>();
  num_scans_ = config["num_scans"].as<int32_t>();
  filter_num_ = config["filter_num"].as<int32_t>();
  voxel_size_ = config["voxel_size"].as<float>();
  min_range_ = config["min_range"].as<float>();
  max_range_ = config["max_range"].as<float>();
  max_iter_times_ = config["max_iter_times"].as<int32_t>();
  meas_cov_ = config["meas_cov"].as<double>();
  std::string time_scale_type = config["time_scale"].as<std::string>();
  if (time_scale_type == "s") {
    time_scale_ = 1.f;
  } else if (time_scale_type == "ms") {
    time_scale_ = 0.001;
  } else if (time_scale_type == "us") {
    time_scale_ = 1e-6;
  } else if (time_scale_type == "ns") {
    time_scale_ = 1e-9;
  }
  std::vector<double> offset_ril = config["offset_ril"].as<std::vector<double>>();
  std::vector<double> offset_til = config["offset_til"].as<std::vector<double>>();
  ril_ << offset_ril.at(0), offset_ril.at(1), offset_ril.at(2),
      offset_ril.at(3), offset_ril.at(4), offset_ril.at(5),
      offset_ril.at(6), offset_ril.at(7), offset_ril.at(8);
  til_ << offset_til.at(0), offset_til.at(1), offset_til.at(2);
  LOG(INFO) << "Init, extrinsic rot_imu_LiDAR: " << std::endl
            << ril_ << std::endl
            << til_;
  pub_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_world", 100000);
  pub_cloud_dense_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_dense", 100000);
  pub_cloud_effect_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effect", 100000);
  LOG(WARNING) << "LidarHandler Init Succ";
}

LidarHandler::~LidarHandler() {
  std::cout << "\033[32mLidarHandler Destruction!\033[0m" << std::endl;
}

Eigen::Vector3f LidarHandler::PointBodyToWorld(const Eigen::Vector3f& point_b,
                                               const Ikfom::IkfomState& state) {
  const Vector3f& point_i = state.offset_ril.rot() * point_b + state.offset_til;
  return state.rot.rot() * point_i + state.pos;
}

Eigen::Vector3f LidarHandler::PointWorldToBody(const Vector3f& point_w,
                                               const Ikfom::IkfomState& state) {
  const Vector3f& point_i = state.rot.rot().conjugate() * (point_w - state.pos);
  return state.offset_ril.rot().conjugate() * (point_i - state.offset_til);
}

bool LidarHandler::CloudBodyToWorld(const LIVOCloud::ConstPtr& pcl_in,
                                    const Ikfom::IkfomState& state,
                                    LIVOCloud* const pcl_out) const {
  pcl_out->resize(pcl_in->size());
  for (uint32_t i = 0; i < static_cast<uint32_t>(pcl_in->size()); ++i) {
    const LIVOPoint& pi = pcl_in->points[i];
    LIVOPoint& po = pcl_out->points[i];
    const Eigen::Vector3f p_body = {pi.x, pi.y, pi.z};
    const Vector3f& p_world = PointBodyToWorld(p_body, state);
    po.x = p_world(0);
    po.y = p_world(1);
    po.z = p_world(2);
    po.intensity = pi.intensity;
  }
  return true;
}

bool LidarHandler::CloudWorldToBody(
    const LIVOCloud::ConstPtr& pcl_in,
    const Ikfom::IkfomState& state,
    LIVOCloud* const pcl_out) const {
  pcl_out->resize(pcl_in->size());
  for (uint32_t i = 0; i < static_cast<uint32_t>(pcl_in->size()); ++i) {
    const LIVOPoint& pi = pcl_in->points[i];
    LIVOPoint& po = pcl_out->points[i];
    const Eigen::Vector3f p_world = {pi.x, pi.y, pi.z};
    const Vector3f& p_body = PointWorldToBody(p_world, state);
    po.x = p_body(0);
    po.y = p_body(1);
    po.z = p_body(2);
    po.intensity = pi.intensity;
  }
  return true;
}

bool LidarHandler::VoxelFilter(const LIVOCloud::ConstPtr& pcl_in,
                               LIVOCloud* const pcl_out) {
  assert(pcl_in != nullptr && pcl_out != nullptr);
  std::unordered_map<VOXEL_KEY, std::vector<LIVOPoint>> grid_map;
  for (const auto& point : pcl_in->points) {
    double loc_xyz[3];
    const Vector3f& pt_w = point.getArray3fMap();
    for (int i = 0; i < 3; i++) {
      loc_xyz[i] = pt_w[i] / voxel_size_;
      if (loc_xyz[i] < 0) {
        loc_xyz[i] -= 1.0;
      }
    }
    VOXEL_KEY key(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
    auto iter = grid_map.find(key);
    if (iter == grid_map.end()) {
      grid_map[key] = std::vector<LIVOPoint>();
    }
    grid_map[key].emplace_back(point);
  }
  pcl_out->reserve(grid_map.size());
  for (auto iter = grid_map.begin(); iter != grid_map.end(); ++iter) {
    const auto& points = iter->second;
    Eigen::Vector3f sum_position(0, 0, 0);
    float sum_intensity = 0.0f;
    for (const auto& pt : points) {
      sum_position += pt.getVector3fMap();
      sum_intensity += pt.intensity;
    }
    LIVOPoint avg_point;
    avg_point.x = static_cast<float>(sum_position.x() / points.size());
    avg_point.y = static_cast<float>(sum_position.y() / points.size());
    avg_point.z = static_cast<float>(sum_position.z() / points.size());
    avg_point.intensity = sum_intensity / points.size();
    pcl_out->points.emplace_back(avg_point);
  }
  pcl_out->width = pcl_out->points.size();
  pcl_out->height = 1;
  pcl_out->is_dense = true;

  return true;
}

LIVOCloud::Ptr LidarHandler::InputScan(const LiDARCloud::ConstPtr& lidar_in,
                                       const std::deque<ImuState>& imustate_vec,
                                       const Ikfom::IkfomState& x) {
  const auto& t1 = omp_get_wtime();
  cur_lidar_time_ = lidar_in->back().time;
  scan_undistort_->clear();
  scan_dense_body_->clear();
  scan_world_->clear();
  scan_body_->clear();
  effect_world_->clear();
  effect_body_->clear();
  scan_undistort_->reserve(lidar_in->size());
  scan_dense_body_->reserve(lidar_in->size());
  if (!Undistortion(lidar_in, x.offset_ril.matrix(),
                    x.offset_til, imustate_vec)) {
    return nullptr;
  }
  CloudWorldToBody(scan_undistort_, x, scan_dense_body_.get());
  VoxelFilter(scan_dense_body_, scan_body_.get());
  CloudBodyToWorld(scan_body_, x, scan_world_.get());
  const auto& t2 = omp_get_wtime();
  t_lid_pre += (t2 - t1) * 1000.;

  LOG(INFO) << "[LiDAR Preprocess]: undistortion and voxel filter cost "
            << (t2 - t1) * 1000. << " ms.";

  return scan_undistort_;
}

bool LidarHandler::Undistortion(const LiDARCloud::ConstPtr& lidar_in,
                                const Eigen::Matrix3f& ril, const Vector3f& til,
                                const std::deque<ImuState>& imustate_vec) {
  assert(lidar_in != nullptr && scan_undistort_ != nullptr);
  if (imustate_vec.empty()) {
    LOG(INFO) << "[Undistortion]: empty imustate_vec";
    return false;
  }
  scan_undistort_->clear();
  scan_undistort_->reserve(lidar_in->size());
  double last_point_time = -1.;
  float uncertainty = 0.f;
  for (int32_t i = 0; i < lidar_in->size(); ++i) {
    const auto& point_in = lidar_in->at(i);
    if (point_in.time < kMathEpsilon) continue;
    for (auto vit = imustate_vec.end() - 2; vit >= imustate_vec.begin(); --vit) {
      if (vit->timestamp > point_in.time && vit != imustate_vec.begin()) {
        continue;
      }
      // point later than pose state
      double dt = point_in.time - vit->timestamp;
      // double dt = std::fmax(point_timestamp - vit->timestamp, 0.);
      const Vector3f& gyr_w = (vit + 1)->gyr_w;
      const Vector3f& acc_w = (vit + 1)->acc_w;
      // compute undistort point at lidar end time (infact nearest imu end time)
      const Vector3f point_l = {point_in.x, point_in.y, point_in.z};
      const Vector3f& point_i = ril * point_l + til;
      const Matrix3f& rwi = vit->rot_w.matrix() * mtk::SO3::exp(gyr_w, dt).matrix();
      const Vector3f& twi = vit->pos_w + vit->vel_w * dt + 0.5 * acc_w * dt * dt;
      const Vector3f& point_w = rwi * point_i + twi;
      LIVOPoint point_out;
      point_out.x = point_w(0);
      point_out.y = point_w(1);
      point_out.z = point_w(2);
      point_out.intensity = point_in.intensity;
      last_point_time = point_in.time;
      scan_undistort_->emplace_back(point_out);
      break;
    }
  }
  if (lidar_in->size() != scan_undistort_->points.size()) {
    std::cout << lidar_in->size() << ' ' << scan_undistort_->points.size() << std::endl;
  }
  return true;
}

bool LidarHandler::Observation(const Ikfom::IkfomState& x, bool knn,
                               Eigen::MatrixXf* const h, Eigen::VectorXf* const r,
                               Eigen::VectorXf* const c) {
  const auto& t1 = omp_get_wtime();
  if (knn) {
    CloudBodyToWorld(scan_body_, x, scan_world_.get());
    knn_surfels_.clear();
    local_map_->GetNearestSurfels(scan_world_, knn_surfels_);
    effect_body_->resize(knn_surfels_.size());
    effect_world_->resize(knn_surfels_.size());
    int i = 0, effect_num = 0;
    for (; i < knn_surfels_.size(); ++i) {
      if (knn_surfels_.at(i) != nullptr) {
        knn_surfels_.at(effect_num) = knn_surfels_.at(i);
        effect_body_->at(effect_num) = scan_body_->at(i);
        effect_world_->at(effect_num) = scan_world_->at(i);
        ++effect_num;
      }
    }
    knn_surfels_.resize(effect_num);
    effect_body_->resize(effect_num);
    effect_world_->resize(effect_num);
    if (effect_num < 5) {
      LOG(WARNING) << "[LiDAR Observation]: no effect point "
                   << effect_num << " / " << scan_world_->size();
      return false;
    }
  }
  const auto& t2 = omp_get_wtime();

  *h = Eigen::MatrixXf::Zero(knn_surfels_.size(), 6);
  r->resize(knn_surfels_.size());
  c->resize(knn_surfels_.size());
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(none) shared(effect_body_, knn_surfels_, h, r, c, x)
#endif
  for (int32_t i = 0; i < effect_world_->size(); ++i) {
    const Vector3f& pt_w = effect_world_->at(i).getVector3fMap();
    const Vector3f& plane_coef = knn_surfels_.at(i)->normal;
    const Eigen::Matrix<float, 1, 3>& d_obs_ptw = plane_coef.transpose();
    const Vector3f& pt_i = x.rot.rot().conjugate() * (pt_w - x.pos);
    const Matrix3f& d_state_error = lie_algebra::RightJacobianInv(x.rot.log());
    h->block<1, 3>(i, 0) = d_obs_ptw * x.rot.matrix() * mtk::SO3::hat(-pt_i);
    h->block<1, 3>(i, 3) = d_obs_ptw;
    (*r)(i) = -(plane_coef.dot(pt_w) + knn_surfels_.at(i)->d);
    (*c)(i) = meas_cov_;
    // (*c)(i) = knn_surfels_.at(i)->sigma[0] * 0.1;
  }

  const auto& t3 = omp_get_wtime();

  t_lid_obs += (t3 - t1) * 1000.;

  LOG(INFO) << "[LiDAR Observation]: knn (" << knn_surfels_.size() << " / "
            << knn_surfels_.size() << ") use time " << (t2 - t1) * 1000. << " ms, "
            << "compute residual cost " << (t3 - t2) * 1000. << " ms.";
  return true;
}

HybridCloud::Ptr LidarHandler::MapIncremental(const Ikfom::IkfomState& x) {
  double t1 = omp_get_wtime();
  CloudBodyToWorld(scan_body_, x, scan_world_.get());
  CloudBodyToWorld(effect_body_, x, effect_world_.get());
  CloudBodyToWorld(scan_dense_body_, x, scan_undistort_.get());

  HybridCloud::Ptr hybrid_cloud{new HybridCloud};
  hybrid_cloud->resize(scan_undistort_->size());
  for (int32_t i = 0; i < hybrid_cloud->size(); ++i) {
    const LIVOPoint& pi = scan_undistort_->at(i);
    HybridPoint& po = hybrid_cloud->at(i);
    po.x = pi.x;
    po.y = pi.y;
    po.z = pi.z;
    po.intensity = pi.intensity;
    po.cnt = 1;
  }
  double t2 = omp_get_wtime();

  LOG(INFO) << "[LiDAR MapIncremental]: Update " << hybrid_cloud->size()
            << " LiDAR Points cost " << (t2 - t1) * 1000. << "ms";

  return hybrid_cloud;
}

void LidarHandler::Publish() {
  if (pub_cloud_dense_.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*scan_undistort_, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(cur_lidar_time_);
    cloud_msg.header.frame_id = "camera_init";
    pub_cloud_dense_.publish(cloud_msg);
  }
  if (pub_cloud_world_.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*scan_world_, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(cur_lidar_time_);
    cloud_msg.header.frame_id = "camera_init";
    pub_cloud_world_.publish(cloud_msg);
  }
  if (pub_cloud_effect_.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*scan_effect_, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(cur_lidar_time_);
    cloud_msg.header.frame_id = "camera_init";
    pub_cloud_effect_.publish(cloud_msg);
  }
  if (pcd_save_en_) {
    *scan_save_ += *scan_undistort_;
  }
}

void LidarHandler::SavePcd() {
  if (pcd_save_en_) {
    std::string save_path = std::string(ROOT_DIR) + "Log/PCD/cloud.pcd";
    LOG(WARNING) << "saving pcd to " << save_path;
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(save_path, *scan_save_);
  }
}

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
