/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file pub_sub.cc
 **/

#include "livo.h"

#include <cmath>
#include <cstring>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <glog/logging.h>
#include <omp.h>

namespace livo {

void LIVO::VelodyneCallBack(const sensor_msgs::PointCloud2::ConstPtr& lidar_in) {
  static double last_timestamp = 0;
  double timestamp = lidar_in->header.stamp.toSec();
  if (timestamp < last_timestamp) {
    LOG(WARNING) << "[lio]: VelodyneCallBack, lidar loop back, clear buffer";
    mtx_buffer_.lock();
  }
  const auto& t1 = std::chrono::steady_clock::now();
  pcl::PointCloud<velodyne_ros::Point> pcl_origin;
  pcl::fromROSMsg(*lidar_in, pcl_origin);
  const auto& t2 = std::chrono::steady_clock::now();
  pcl::PointCloud<velodyne_ros::Point> pcl_filter;
  pcl_filter.reserve(pcl_origin.size() / lidar_handler_->filter_num());
  for (int32_t i = 0; i < pcl_origin.size(); ++i) {
    if ((i) % lidar_handler_->filter_num() != 0) continue;
    if (std::hypot(pcl_origin.points.at(i).x, pcl_origin.points.at(i).y,
                   pcl_origin.points.at(i).z) < lidar_handler_->min_range()) {
      continue;
    }
    pcl_filter.points.emplace_back(pcl_origin.points.at(i));
  }
  const auto& t3 = std::chrono::steady_clock::now();
  std::sort(pcl_filter.points.begin(), pcl_filter.points.end(),
            [](const velodyne_ros::Point& a,
               const velodyne_ros::Point& b) { return a.time < b.time; });
  const auto& t4 = std::chrono::steady_clock::now();
  float time_base = pcl_filter.points.back().time;
  for (int32_t i = 0; i < pcl_filter.size(); ++i) {
    const auto& point_in = pcl_filter.points.at(i);
    LiDARPoint pcl_point;
    pcl_point.x = point_in.x;
    pcl_point.y = point_in.y;
    pcl_point.z = point_in.z;
    pcl_point.intensity = point_in.intensity;
    pcl_point.time = timestamp + lidar_handler_->time_scale() *
                                     float(point_in.time);
    pcl_point.timebase = timestamp + lidar_handler_->time_scale() *
                                         time_base;
    pcl_point.is_end = 0;
    point_buffer_.emplace_back(pcl_point);
  }
  point_buffer_.back().is_end = 1;
  const auto& t5 = std::chrono::steady_clock::now();
  mtx_buffer_.lock();
  last_timestamp = timestamp;
  LOG(INFO) << "VelodyneCallBack: receive lidar frame at:" << std::fixed
            << lidar_in->header.stamp.toSec() << ", new timebase: " << float(time_base)
            << ", convert use:" << std::chrono::duration<double>(t5 - t4).count();
  mtx_buffer_.unlock();
  sig_buffer_.notify_all();
}

void LIVO::OusterCallBack(const sensor_msgs::PointCloud2::ConstPtr& lidar_in) {
  static double last_timestamp = 0;
  static int32_t filter_bias = 0;
  // filter_bias = (filter_bias + 1) % lidar_handler_->filter_num();
  double timestamp = lidar_in->header.stamp.toSec();
  if (timestamp < last_timestamp) {
    LOG(WARNING) << "[lio]: OusterCallBack, lidar loop back, clear buffer";
  }
  const auto& t1 = omp_get_wtime();
  pcl::PointCloud<ouster_ros::Point> pcl_origin;
  pcl::fromROSMsg(*lidar_in, pcl_origin);
  const auto& t2 = omp_get_wtime();
  pcl::PointCloud<ouster_ros::Point> pcl_filter;

  pcl_filter.reserve(pcl_origin.size() / lidar_handler_->filter_num());
  for (int32_t i = 0; i < pcl_origin.size(); ++i) {
    if ((i + filter_bias) % lidar_handler_->filter_num() != 0) continue;
    if (std::hypot(pcl_origin.points.at(i).x, pcl_origin.points.at(i).y,
                   pcl_origin.points.at(i).z) < lidar_handler_->min_range()) {
      continue;
    }
    pcl_filter.points.emplace_back(pcl_origin.points.at(i));
  }
  const auto& t3 = omp_get_wtime();
  std::sort(pcl_filter.points.begin(), pcl_filter.points.end(),
            [](const ouster_ros::Point& a,
               const ouster_ros::Point& b) { return a.t < b.t; });
  const auto& t4 = omp_get_wtime();
  uint32_t time_base = pcl_filter.points.back().t;
  // seperate frame
  for (int32_t i = 0; i < pcl_filter.size(); ++i) {
    const auto& point_in = pcl_filter.points.at(i);
    LiDARPoint pcl_point;
    pcl_point.x = point_in.x;
    pcl_point.y = point_in.y;
    pcl_point.z = point_in.z;
    pcl_point.intensity = point_in.reflectivity;
    pcl_point.time = timestamp + lidar_handler_->time_scale() *
                                     (float(point_in.t) - float(time_base));
    pcl_point.timebase = timestamp;
    pcl_point.is_end = 0;
    point_buffer_.emplace_back(pcl_point);
  }
  point_buffer_.back().is_end = 1;
  const auto& t5 = omp_get_wtime();
  mtx_buffer_.lock();
  last_timestamp = timestamp;
  LOG(INFO) << "OusterCallBack: receive lidar frame at:" << std::fixed
            << lidar_in->header.stamp.toSec() << ", new timebase:"
            << " convert use:" << (t5 - t4);
  mtx_buffer_.unlock();
  sig_buffer_.notify_all();
}

void LIVO::LivoxCallBack(const CustomMsg::ConstPtr& lidar_in) {
  static double last_timestamp = 0;
  static double last_timebase = 0;
  double timestamp = lidar_in->header.stamp.toSec();
  if (timestamp < last_timestamp) {
    LOG(WARNING) << "[lio]: LivoxCallBack, lidar loop back, clear buffer";
  }
  auto t1 = omp_get_wtime();
  uint32_t time_base = 0u;
  std::deque<LiDARPoint> point_buffer_tmp;
  int32_t filter_idx = -1;
  for (auto it = lidar_in->points.end() - 1; it >= lidar_in->points.begin(); --it, ++filter_idx) {
    const auto& point_in = *it;
    if (point_in.line > lidar_handler_->num_scans() || point_in.tag & 0x20 != 0x00 ||
        std::hypot(point_in.x, point_in.y, point_in.z) <= lidar_handler_->min_range()) {
      continue;
    }
    if (filter_idx % lidar_handler_->filter_num() != 0) continue;
    if (time_base == 0u) {
      time_base = point_in.offset_time;
    }
    LiDARPoint pcl_point;
    pcl_point.x = point_in.x;
    pcl_point.y = point_in.y;
    pcl_point.z = point_in.z;
    pcl_point.intensity = point_in.reflectivity;

    pcl_point.time = timestamp + lidar_handler_->time_scale() *
                                     float(point_in.offset_time);
    pcl_point.timebase = timestamp + lidar_handler_->time_scale() *
                                         float(time_base);
    pcl_point.is_end = 0;
    if (pcl_point.time <= last_timebase) pcl_point.time = point_buffer_tmp.front().time;
    point_buffer_tmp.emplace_front(pcl_point);
  }
  point_buffer_tmp.back().is_end = 1;
  mtx_buffer_.lock();
  for (auto it = point_buffer_tmp.begin(); it != point_buffer_tmp.end(); ++it) {
    point_buffer_.emplace_back(*it);
  }
  auto t2 = omp_get_wtime();
  double duration = (t2 - t1);
  last_timestamp = timestamp;
  last_timebase = point_buffer_.back().timebase;
  LOG(INFO) << std::fixed << "LivoxCallBack: receive "
            << point_buffer_tmp.size() << " / " << lidar_in->points.size()
            << "points , from " << point_buffer_tmp.front().time
            << " to " << point_buffer_tmp.back().time
            << ", new timebase:" << point_buffer_.back().timebase
            << ", convert use:" << duration;
  mtx_buffer_.unlock();
  sig_buffer_.notify_all();
}

void LIVO::ImuCallBack(const sensor_msgs::Imu::ConstPtr& imu_in) {
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*imu_in));
  msg->header.stamp.fromSec(imu_in->header.stamp.toSec() + imu_time_offset_);
  static double last_timestamp = 0;
  double timestamp = msg->header.stamp.toSec();
  const Eigen::Vector3f cur_acc = {float(msg->linear_acceleration.x),
                                   float(msg->linear_acceleration.y),
                                   float(msg->linear_acceleration.z)};
  mtx_buffer_.lock();
  if (timestamp < last_timestamp) {
    LOG(WARNING) << "[livo]: ImuCallBack, imu loop back, clear buffer";
    imu_buffer_.clear();
  }
  last_timestamp = timestamp;
  imu_buffer_.emplace_back(msg);
  mtx_buffer_.unlock();
}

void LIVO::ImageCallBack(const sensor_msgs::Image::ConstPtr& img_in) {
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*img_in));
  static double last_timestamp = 0.;
  static int32_t filter = -1;
  ++filter;
  // if (filter % 3 != 0) return;
  msg->header.stamp.fromSec(img_in->header.stamp.toSec() + img_time_offset_);
  double timestamp = msg->header.stamp.toSec();
  mtx_buffer_.lock();
  if (timestamp < last_timestamp) {
    LOG(WARNING) << "[livo]: ImageCallBack, img loop back, clear buffer";
    img_buffer_.clear();
  }
  last_timestamp = timestamp;
  img_buffer_.emplace_back(msg);
  mtx_buffer_.unlock();
  LOG(INFO) << "ImageCallBack: receive image frame at:" << std::fixed
            << timestamp;
}

void LIVO::PublishPath() {
  geometry_msgs::PoseStamped msg_body_pose;
  msg_body_pose.pose.position.x = x_.pos(0);
  msg_body_pose.pose.position.y = x_.pos(1);
  msg_body_pose.pose.position.z = x_.pos(2);
  msg_body_pose.pose.orientation.x = x_.rot.rot().coeffs()[0];
  msg_body_pose.pose.orientation.y = x_.rot.rot().coeffs()[1];
  msg_body_pose.pose.orientation.z = x_.rot.rot().coeffs()[2];
  msg_body_pose.pose.orientation.w = x_.rot.rot().coeffs()[3];
  msg_body_pose.header.stamp = ros::Time().fromSec(cur_sensor_timestamp_);
  msg_body_pose.header.frame_id = "camera_init";

  path_.poses.push_back(msg_body_pose);
  path_.header.frame_id = "camera_init";
  path_.header.stamp = ros::Time().fromSec(cur_sensor_timestamp_);
  pub_path_.publish(path_);
}

void LIVO::PublishOdometry() {
  if (pub_odometry_.getNumSubscribers() > 0) {
    nav_msgs::Odometry odom;
    odom.header.frame_id = "camera_init";
    odom.child_frame_id = "body";
    odom.header.stamp = ros::Time().fromSec(cur_sensor_timestamp_);
    odom.pose.pose.position.x = x_.pos.x();
    odom.pose.pose.position.y = x_.pos.y();
    odom.pose.pose.position.z = x_.pos.z();
    odom.pose.pose.orientation.x = x_.rot.rot().x();
    odom.pose.pose.orientation.y = x_.rot.rot().y();
    odom.pose.pose.orientation.z = x_.rot.rot().z();
    odom.pose.pose.orientation.w = x_.rot.rot().w();
    pub_odometry_.publish(odom);

    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = odom.header.stamp;
    transformStamped.header.frame_id = "camera_init";
    transformStamped.child_frame_id = "body";
    transformStamped.transform.translation.x = odom.pose.pose.position.x;
    transformStamped.transform.translation.y = odom.pose.pose.position.y;
    transformStamped.transform.translation.z = odom.pose.pose.position.z;
    transformStamped.transform.rotation.x = odom.pose.pose.orientation.x;
    transformStamped.transform.rotation.y = odom.pose.pose.orientation.y;
    transformStamped.transform.rotation.z = odom.pose.pose.orientation.z;
    transformStamped.transform.rotation.w = odom.pose.pose.orientation.w;
    br.sendTransform(transformStamped);
  }
}

void LIVO::PublishHybridMap() {
  static int32_t pub_interval = 0;
  if (pub_interval++ % 10 != 0) return;

  if (pub_map_points_.getNumSubscribers() > 0) {
    ColorCloud::Ptr color_points = map_->FlattenPoints();
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*color_points, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(cur_sensor_timestamp_);
    cloud_msg.header.frame_id = "camera_init";
    pub_map_points_.publish(cloud_msg);
  }

  if (pub_map_colorpoints_.getNumSubscribers() > 0) {
    ColorCloud::Ptr map_points = map_->FlattenPoints();
    ColorCloud::Ptr color_points{new ColorCloud};
    color_points->reserve(map_points->size());
    for (const auto& point : map_points->points) {
      pcl::PointXYZRGB color_point;
      if (point.r == 0 || point.g == 0 || point.b == 0) continue;
      color_point.x = point.x;
      color_point.y = point.y;
      color_point.z = point.z;
      color_point.r = point.r;
      color_point.g = point.g;
      color_point.b = point.b;
      color_points->points.emplace_back(color_point);
    }
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*color_points, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(cur_sensor_timestamp_);
    cloud_msg.header.frame_id = "camera_init";
    pub_map_colorpoints_.publish(cloud_msg);
  }

  if (pub_map_feats_.getNumSubscribers() > 0) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_points{
        new pcl::PointCloud<pcl::PointXYZRGB>};
    pcl::copyPointCloud(*map_->FlattenVisFeats(), *color_points);
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*color_points, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(cur_sensor_timestamp_);
    cloud_msg.header.frame_id = "camera_init";
    pub_map_feats_.publish(cloud_msg);
  }

  if (pub_map_spheres_.getNumSubscribers() > 0) {
    visualization_msgs::MarkerArray spheres_msg;
    const auto& spheres = map_->FlattenSurfels();
    int32_t i = 0;
    for (const auto& sphere : spheres) {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = ros::Time().fromSec(cur_sensor_timestamp_);
      marker.ns = "spheres";
      marker.id = i++;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = sphere->center.x();
      marker.pose.position.y = sphere->center.y();
      marker.pose.position.z = sphere->center.z();
      marker.pose.orientation.x = sphere->rotation.x();
      marker.pose.orientation.y = sphere->rotation.y();
      marker.pose.orientation.z = sphere->rotation.z();
      marker.pose.orientation.w = sphere->rotation.w();
      marker.scale.x = sphere->sigma[2] * 3;
      marker.scale.y = sphere->sigma[1] * 3;
      marker.scale.z = sphere->sigma[0] * 3;
      marker.color.r = sphere->color[0] / 255.f;
      marker.color.g = sphere->color[1] / 255.f;
      marker.color.b = sphere->color[2] / 255.f;
      marker.color.a = 0.8f;
      spheres_msg.markers.emplace_back(marker);

      marker.ns = "centers";
      marker.id = i;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = sphere->center.x();
      marker.pose.position.y = sphere->center.y();
      marker.pose.position.z = sphere->center.z();
      marker.pose.orientation.x = sphere->rotation.x();
      marker.pose.orientation.y = sphere->rotation.y();
      marker.pose.orientation.z = sphere->rotation.z();
      marker.pose.orientation.w = sphere->rotation.w();
      marker.scale.x = 0.1;
      marker.scale.y = 0.1;
      marker.scale.z = 0.1;
      marker.color.r = 1.f;
      marker.color.g = 1.f;
      marker.color.b = 0.f;
      marker.color.a = 1.0f;
      spheres_msg.markers.emplace_back(marker);
    }
    pub_map_spheres_.publish(spheres_msg);
  }

  if (pub_map_voxels_.getNumSubscribers() > 0) {
    visualization_msgs::MarkerArray voxels_msg;
    VoxelCloud::Ptr voxels = map_->FlattenMapVoxel();
    std::vector<int32_t> i;
    for (const auto& voxel : voxels->points) {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = ros::Time().fromSec(cur_sensor_timestamp_);
      marker.ns = "layer_" + std::to_string(voxel.vox_layer);
      while (voxel.vox_layer + 1 > i.size()) i.emplace_back(0);
      marker.id = i.at(voxel.vox_layer)++;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = voxel.x;
      marker.pose.position.y = voxel.y;
      marker.pose.position.z = voxel.z;
      marker.pose.orientation.x = 0.;
      marker.pose.orientation.y = 0.;
      marker.pose.orientation.z = 0.;
      marker.pose.orientation.w = 1.;
      marker.scale.x = voxel.vox_size;
      marker.scale.y = voxel.vox_size;
      marker.scale.z = voxel.vox_size;
      float color[3] = {0.f, 0.f, 0.f};
      switch (voxel.vox_layer) {
        case 0:
          color[0] = 203.f;
          color[1] = 224.f;
          color[2] = 187.f;
          break;
        case 1:
          color[0] = 255.f;
          color[1] = 160.f;
          color[2] = 160.f;
          break;
        case 2:
          color[0] = 120.f;
          color[1] = 200.f;
          color[2] = 255.f;
          break;
        case 3:
          color[0] = 160.f;
          color[1] = 200.f;
          color[2] = 120.f;
          break;

        default:
          break;
      }
      marker.color.r = color[0] / 255.f;
      marker.color.g = color[1] / 255.f;
      marker.color.b = color[2] / 255.f;
      marker.color.a = voxel.layer_alpha;
      voxels_msg.markers.push_back(marker);
    }
    pub_map_voxels_.publish(voxels_msg);
  }
}

}  // namespace livo