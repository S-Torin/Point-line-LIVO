/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file livo.h
 **/

#pragma once

#include <deque>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <condition_variable>

#include <livo/CustomMsg.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <yaml-cpp/yaml.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "common/ikfom.h"
#include "map/hybrid_map.h"
#include "point_type.h"
#include "sensor/imu/imu_handler.h"
#include "sensor/lidar/lidar_handler.h"
#include "sensor/cam/cam_handler.h"

namespace livo {

class LIVO {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  explicit LIVO(const YAML::Node& config, ros::NodeHandle& nh);
  ~LIVO();
  void Process();

 private:
  struct SyncPackage {
    LiDARCloud::Ptr points_msg = nullptr;
    std::deque<ImuState> imu_states;
    std::vector<sensor_msgs::Imu::ConstPtr> imu_msgs;
    sensor_msgs::Image::ConstPtr img_msg = nullptr;
    bool first_lidar = true;
    void clear();
    SyncPackage() { clear(); }
  };

 private:
  bool SynchronousInputMsgs();
  bool IeskfProcess();
  bool IeskfUpdate(const Eigen::MatrixXf& h, const Eigen::VectorXf& r,
                   const Eigen::VectorXf& c);
  void VelodyneCallBack(const sensor_msgs::PointCloud2::ConstPtr& lidar_in);
  void OusterCallBack(const sensor_msgs::PointCloud2::ConstPtr& lidar_in);
  void LivoxCallBack(const CustomMsg::ConstPtr& lidar_in);
  void ImuCallBack(const sensor_msgs::Imu::ConstPtr& imu_in);
  void ImageCallBack(const sensor_msgs::Image::ConstPtr& img_in);

  void PublishPath();
  void PublishOdometry();
  void PublishHybridMap();

 private:
  double cur_sensor_timestamp_ = 0.;
  double init_distance_ = 0.5;
  double imu_time_offset_ = 0.;
  double img_time_offset_ = 0.;
  bool start_visual_ = false;
  bool imu_initialized_ = false;
  bool first_package_ = true;
  int32_t map_init_iter_ = 0;
  int32_t map_init_times_ = 0;

  std::shared_ptr<HybridMap> map_ = nullptr;
  std::shared_ptr<ImuHandler> imu_handler_ = nullptr;
  std::shared_ptr<LidarHandler> lidar_handler_ = nullptr;
  std::shared_ptr<CamHandler> cam_handler_ = nullptr;
  SyncPackage sync_package_;

  Ikfom::IkfomState x_;
  Ikfom::IkfomCovMat cov_ = Ikfom::IkfomCovMat::Identity();
  Ikfom::IkfomState predict_x_;
  Ikfom::IkfomCovMat predict_cov_ = Ikfom::IkfomCovMat::Identity();
  HybridCloud::Ptr scan_hybrid_{new HybridCloud};

  std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;
  std::deque<sensor_msgs::PointCloud::ConstPtr> lidar_buffer_;
  std::deque<sensor_msgs::Image::ConstPtr> img_buffer_;
  std::deque<LiDARPoint> point_buffer_;

  std::mutex mtx_buffer_;
  std::condition_variable sig_buffer_;
  ros::Subscriber sub_lidar_;
  ros::Subscriber sub_imu_;
  ros::Subscriber sub_img_;
  ros::Publisher pub_map_points_;
  ros::Publisher pub_map_colorpoints_;
  ros::Publisher pub_map_feats_;
  ros::Publisher pub_map_spheres_;
  ros::Publisher pub_map_voxels_;
  ros::Publisher pub_odometry_;
  ros::Publisher pub_path_;
  nav_msgs::Path path_;
};

}  // namespace livo
