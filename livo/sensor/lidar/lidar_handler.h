/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file lidar_handler.h
 **/

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <livo/CustomMsg.h>
#include <pcl/common/common.h>
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <sensor_msgs/PointCloud.h>
#include <yaml-cpp/yaml.h>

#include "common/ikfom.h"
#include "map/hybrid_map.h"
#include "point_type.h"

namespace livo {
class LidarHandler final {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit LidarHandler(const YAML::Node& lidar_handler_config,
                        ros::NodeHandle& nh, std::shared_ptr<HybridMap> local_map);
  ~LidarHandler();
  LIVOCloud::Ptr InputScan(const LiDARCloud::ConstPtr& lidar_in,
                           const std::deque<ImuState>& imustate_vec,
                           const Ikfom::IkfomState& x);
  bool Observation(const Ikfom::IkfomState& x, bool knn,
                   Eigen::MatrixXf* const h, Eigen::VectorXf* const r,
                   Eigen::VectorXf* const c);
  HybridCloud::Ptr MapIncremental(const Ikfom::IkfomState& x);
  void Publish();
  void SavePcd();

  bool CloudBodyToWorld(const LIVOCloud::ConstPtr& pcl_in, const Ikfom::IkfomState& state,
                        LIVOCloud* const pcl_out) const;
  bool CloudWorldToBody(const LIVOCloud::ConstPtr& pcl_in, const Ikfom::IkfomState& state,
                        LIVOCloud* const pcl_out) const;

  LIVOCloud::ConstPtr scan_undistort() const { return scan_undistort_; }
  LIVOCloud::ConstPtr scan_world() const { return scan_world_; }
  float time_scale() const { return time_scale_; }
  int32_t filter_num() const { return filter_num_; }
  int32_t num_scans() const { return num_scans_; }
  int32_t iter_times() const { return max_iter_times_; }
  double min_range() const { return min_range_; }
  std::string lidar_type() const { return lidar_type_; }
  Eigen::Matrix3f ril() const { return ril_; }
  Eigen::Vector3f til() const { return til_; }
  double meas_cov() const { return meas_cov_; }

 private:
  static Eigen::Vector3f PointWorldToBody(const Eigen::Vector3f& point_w,
                                          const Ikfom::IkfomState& state);
  static Eigen::Vector3f PointBodyToWorld(const Eigen::Vector3f& point_b,
                                          const Ikfom::IkfomState& state);
  bool VoxelFilter(const LIVOCloud::ConstPtr& pcl_in, LIVOCloud* const pcl_out);
  bool Undistortion(const LiDARCloud::ConstPtr& lidar_in,
                    const Eigen::Matrix3f& ril, const Eigen::Vector3f& til,
                    const std::deque<ImuState>& imustate_vec);

 private:
  bool enable_ = true;
  bool pcd_save_en_ = false;
  std::string lidar_type_ = "velodyne";
  int32_t num_scans_ = 32;
  int32_t filter_num_ = 2;
  double cur_lidar_time_ = 0.;
  double time_scale_ = 0.001;
  double min_range_ = 2.;
  double max_range_ = 200.;
  double voxel_size_ = 0.5;
  Eigen::Matrix3f ril_ = Eigen::Matrix3f::Identity();
  Eigen::Vector3f til_ = Eigen::Vector3f::Zero();
  double meas_cov_ = 0.001;
  int32_t max_iter_times_ = 3;

  std::shared_ptr<HybridMap> local_map_ = nullptr;

  LIVOCloud::Ptr scan_undistort_{new LIVOCloud};
  LIVOCloud::Ptr scan_dense_body_{new LIVOCloud};
  LIVOCloud::Ptr scan_body_{new LIVOCloud};
  LIVOCloud::Ptr scan_world_{new LIVOCloud};
  LIVOCloud::Ptr scan_effect_{new LIVOCloud};
  LIVOCloud::Ptr scan_save_{new LIVOCloud};
  LIVOCloud::Ptr effect_world_{new LIVOCloud};
  LIVOCloud::Ptr effect_body_{new LIVOCloud};
  std::vector<VoxelSurfelConstPtr> knn_surfels_;

  ros::Publisher pub_cloud_world_;
  ros::Publisher pub_cloud_dense_;
  ros::Publisher pub_cloud_effect_;
};

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
