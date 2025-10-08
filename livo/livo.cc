/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file livo.cc
 **/

#include "livo.h"

#include <fstream>
#include <iomanip>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <omp.h>

#include "common/mtk/lie_algebra.h"
#include "common/mtk/so3.h"
#include "log.h"

namespace livo {

int32_t log_pos_index = 0;

double t_imu_pre = 0.;
double t_lid_pre = 0.;
double t_vis_pre = 0.;
double t_lid_obs = 0.;
double t_vis_obs = 0.;
double t_ieskf = 0.;
double t_map_inc = 0.;
double t_total = 0.;

namespace {
constexpr double kMathEpsilon = 1.0e-8;
}  // namespace

void LIVO::SyncPackage::clear() {
  imu_msgs.clear();
  points_msg = nullptr;
  img_msg = nullptr;
}

LIVO::LIVO(const YAML::Node& config, ros::NodeHandle& nh) {
  std::string lidar_topic = config["common"]["lid_topic"].as<std::string>();
  std::string imu_topic = config["common"]["imu_topic"].as<std::string>();
  std::string img_topic = config["common"]["img_topic"].as<std::string>();
  imu_time_offset_ = config["common"]["imu_time_offset"].as<double>(0.);
  img_time_offset_ = config["common"]["img_time_offset"].as<double>(0.);
  map_init_times_ = config["hybrid_map"]["init_iter_times"].as<int32_t>(5);

  map_.reset(new HybridMap(config["hybrid_map"]));
  imu_handler_.reset(new ImuHandler(config["imu_handler"]));
  lidar_handler_.reset(new LidarHandler(config["lidar_handler"], nh, map_));
  if (!img_topic.empty()) {
    cam_handler_.reset(new CamHandler(config["cam_handler"], nh, map_));
  }

  sync_package_.clear();
  x_.rot = mtk::SO3(Eigen::Quaternionf::Identity());
  x_.pos = Eigen::Vector3f::Zero();
  x_.vel = Eigen::Vector3f::Zero();
  x_.bg = Eigen::Vector3f::Zero();
  x_.ba = Eigen::Vector3f::Zero();
  x_.offset_ril = mtk::SO3(lidar_handler_->ril());
  x_.offset_til = lidar_handler_->til();

  cov_.setZero();
  cov_.diagonal().setOnes();
  const auto livo_config = config["livo"];
  cov_.block<3, 3>(0, 0) *= livo_config["init_cov_rot"].as<double>();
  cov_.block<3, 3>(3, 3) *= livo_config["init_cov_pos"].as<double>();
  cov_.block<3, 3>(6, 6) *= livo_config["init_cov_vel"].as<double>();
  cov_.block<3, 3>(9, 9) *= livo_config["init_cov_bg"].as<double>();
  cov_.block<3, 3>(12, 12) *= livo_config["init_cov_ba"].as<double>();
  cov_.block<2, 2>(15, 15) *= livo_config["init_cov_g"].as<double>();
  cov_.block<3, 3>(17, 17) *= livo_config["init_cov_r_i_l"].as<double>();
  cov_.block<3, 3>(20, 20) *= livo_config["init_cov_t_i_l"].as<double>();

  if (lidar_handler_->lidar_type() == "velodyne") {
    sub_lidar_ = nh.subscribe<sensor_msgs::PointCloud2>(
        lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr& msg) {
          VelodyneCallBack(msg);
        });
  } else if (lidar_handler_->lidar_type() == "ouster") {
    sub_lidar_ = nh.subscribe<sensor_msgs::PointCloud2>(
        lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr& msg) {
          OusterCallBack(msg);
        });
  } else if (lidar_handler_->lidar_type() == "livox") {
    sub_lidar_ = nh.subscribe<CustomMsg>(
        lidar_topic, 200000, [this](const CustomMsg::ConstPtr& msg) {
          LivoxCallBack(msg);
        });
  }
  sub_imu_ = nh.subscribe<sensor_msgs::Imu>(
      imu_topic, 200000, [this](const sensor_msgs::Imu::ConstPtr& msg) {
        ImuCallBack(msg);
      });
  sub_img_ = nh.subscribe<sensor_msgs::Image>(
      img_topic, 200000, [this](const sensor_msgs::Image::ConstPtr& msg) {
        ImageCallBack(msg);
      });

  pub_map_points_ = nh.advertise<sensor_msgs::PointCloud2>("/map_points", 100000);
  pub_map_colorpoints_ = nh.advertise<sensor_msgs::PointCloud2>("/map_colorpoints", 100000);
  pub_map_feats_ = nh.advertise<sensor_msgs::PointCloud2>("/map_feats", 100000);
  pub_map_spheres_ = nh.advertise<visualization_msgs::MarkerArray>("/map_spheres", 100000);
  pub_map_voxels_ = nh.advertise<visualization_msgs::MarkerArray>("/map_voxels", 100000);
  pub_odometry_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
  pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);

  std::ofstream fout;
  fout.open(std::string(ROOT_DIR) + "Log/traj.txt");
  fout << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
  fout.close();
  fout.open(std::string(ROOT_DIR) + "Log/timelog.csv");
  fout << "imu_pre, lid_pre, cam_pre, vis_obs, lid_obs, ieskf, map_inc, total"
       << std::endl;
  fout.close();

  LOG(WARNING) << "LIV-GSM Init Succ";
}

LIVO::~LIVO() {
  if (cam_handler_) cam_handler_->SavePcd();
  lidar_handler_->SavePcd();
  std::cout << "\033[32mLIVO Destruction!\033[0m" << std::endl;
};

void LIVO::Process() {
  if (SynchronousInputMsgs()) {
    t_imu_pre = 0.;
    t_lid_pre = 0.;
    t_vis_pre = 0.;
    t_vis_obs = 0.;
    t_lid_obs = 0.;
    t_ieskf = 0.;
    t_map_inc = 0.;
    t_total = 0.;

    double t0 = omp_get_wtime();
    IeskfProcess();
    double t1 = omp_get_wtime();

    LOG(WARNING) << "[LIVO Process] package " << std::fixed << std::setprecision(6)
                 << cur_sensor_timestamp_ << " cost " << (t1 - t0) * 1000. << " ms.";

    std::ofstream fout;
    fout.open(std::string(ROOT_DIR) + "Log/traj.txt", std::ios::app);
    fout << std::fixed << std::setprecision(6) << cur_sensor_timestamp_ << " "
         << x_.pos.x() << " " << x_.pos.y() << " " << x_.pos.z() << " "
         << x_.rot.rot().coeffs().x() << " "
         << x_.rot.rot().coeffs().y() << " "
         << x_.rot.rot().coeffs().z() << " "
         << x_.rot.rot().coeffs().w() << std::endl;
    fout.close();

    fout.open(std::string(ROOT_DIR) + "Log/timelog.csv", std::ios::app);
    t_total = t_imu_pre + t_lid_pre + t_vis_pre + t_vis_obs + t_lid_obs +
              t_ieskf + t_map_inc;
    fout << t_imu_pre << "," << t_lid_pre << "," << t_vis_pre << ","
         << t_vis_obs << "," << t_lid_obs << "," << t_ieskf << ","
         << t_map_inc << "," << t_total << std::endl;
    fout.close();
  }
}

bool LIVO::SynchronousInputMsgs() {
  std::lock_guard<std::mutex> lock(mtx_buffer_);
  static bool first_imu = true;
  static bool first_cam = true;
  double lidar_msg_time = -1.;
  double img_msg_time = -1.;
  double update_time = -1;
  if (!sync_package_.imu_msgs.empty() || sync_package_.points_msg != nullptr ||
      sync_package_.img_msg != nullptr) {
    return false;
  }
  // process first imu
  if (imu_buffer_.empty()) {
    return false;
  } else if (point_buffer_.empty()) {
    return false;
  } else if (first_imu) {
    while (!point_buffer_.empty()) {
      if (point_buffer_.front().time >
          imu_buffer_.front()->header.stamp.toSec()) {
        first_imu = false;
        break;
      }
      point_buffer_.pop_front();
    }
    if (point_buffer_.empty()) return false;
  }
  // process cam
  if (sync_package_.first_lidar || cam_handler_ == nullptr) {
    lidar_msg_time = point_buffer_.front().timebase;
    if (imu_buffer_.back()->header.stamp.toSec() < lidar_msg_time) {
      sync_package_.clear();
      return false;
    }
    update_time = lidar_msg_time;
    sync_package_.first_lidar = false;
  } else {
    if (first_cam) {
      while (!img_buffer_.empty()) {
        if (img_buffer_.front()->header.stamp.toSec() > point_buffer_.front().time) {
          first_cam = false;
          break;
        }
        img_buffer_.pop_front();
      }
    }
    if (img_buffer_.empty()) {
      sync_package_.clear();
      return false;
    }
    img_msg_time = img_buffer_.front()->header.stamp.toSec();
    if (imu_buffer_.back()->header.stamp.toSec() < img_msg_time ||
        point_buffer_.back().time < img_msg_time) {
      sync_package_.clear();
      return false;
    }
    sync_package_.img_msg = img_buffer_.front();
    img_buffer_.pop_front();
    update_time = img_msg_time;
  }
  // process lidar
  assert(update_time > 0.);
  sync_package_.points_msg.reset(new LiDARCloud);
  while (!point_buffer_.empty()) {
    if (point_buffer_.front().time > update_time) {
      break;
    }
    sync_package_.points_msg->emplace_back(point_buffer_.front());
    point_buffer_.pop_front();
  }
  if (img_msg_time < 0.) {
    update_time = sync_package_.points_msg->back().time;
  }
  // process imu
  while (!imu_buffer_.empty()) {
    sync_package_.imu_msgs.emplace_back(imu_buffer_.front());
    if (imu_buffer_.front()->header.stamp.toSec() >= update_time) {
      break;
    }
    imu_buffer_.pop_front();
  }

  if (sync_package_.points_msg->empty()) {
    sync_package_.clear();
    return false;
  }
  LOG(INFO) << "SynchronousInputMsgs, " << std::fixed << sync_package_.points_msg->back().time
            << ", [image]: " << img_msg_time
            << ", [update]: " << update_time
            << ", [points]: " << sync_package_.points_msg->size()
            << ", [imu]: " << sync_package_.imu_msgs.size()
            << ", [imus]: " << sync_package_.imu_msgs.front()->header.stamp.toSec()
            << " -> " << sync_package_.imu_msgs.back()->header.stamp.toSec();
  return true;
}

bool LIVO::IeskfProcess() {
  if (sync_package_.imu_msgs.empty()) return false;
  if (first_package_) {
    first_package_ = false;
    sync_package_.clear();
    return false;
  }
  if (!start_visual_) {
    LiDARCloud::Ptr filter_points{new LiDARCloud()};
    filter_points->reserve(sync_package_.points_msg->size());
    for (const auto& pt : sync_package_.points_msg->points) {
      if (std::hypot(pt.x, pt.y, pt.z) < 1.5) continue;
      filter_points->emplace_back(pt);
    }
    sync_package_.points_msg = filter_points;
    if (x_.pos.norm() > init_distance_) start_visual_ = true;
  }

  double update_time = sync_package_.img_msg == nullptr
                           ? sync_package_.points_msg->back().time
                           : sync_package_.img_msg->header.stamp.toSec();
  for (int32_t i = 0; i < sync_package_.imu_msgs.size(); ++i) {
    const auto& cur_imu_msg = sync_package_.imu_msgs.at(i);
    const double imu_msg_time = cur_imu_msg->header.stamp.toSec();
    if (!imu_handler_->ImuInit(cur_imu_msg, &x_, &cov_)) {
      sync_package_.imu_states.clear();
      cur_sensor_timestamp_ = imu_msg_time;
      imu_handler_->InsertImuState(cur_imu_msg, x_, cov_, imu_msg_time,
                                   &sync_package_.imu_states);
    } else if (imu_msg_time <= update_time) {
      cur_sensor_timestamp_ = imu_msg_time;
      imu_handler_->ImuPredict(imu_msg_time, true, cur_imu_msg, &x_,
                               &cov_, &sync_package_.imu_states);
      LOG(INFO) << "  [imu propagation]: " << std::fixed << imu_msg_time;
    } else {
      cur_sensor_timestamp_ = update_time;
      imu_handler_->ImuPredict(update_time, false, cur_imu_msg, &x_,
                               &cov_, &sync_package_.imu_states);
      LOG(INFO) << "  [ieskd update]: " << std::fixed << update_time;

      lidar_handler_->InputScan(sync_package_.points_msg, sync_package_.imu_states, x_);
      if (cam_handler_) {
        cam_handler_->InputImage(sync_package_.img_msg, lidar_handler_->scan_world(), x_);
      }
      if (map_->empty() || map_init_iter_ < map_init_times_) {
        HybridCloud::Ptr hybrid_cloud = lidar_handler_->MapIncremental(x_);
        if (cam_handler_ != nullptr) cam_handler_->RenderScan(x_, hybrid_cloud);
        map_->InsertPoints(hybrid_cloud);
        ++map_init_iter_;
        ++i;
        continue;
      } else {
        sync_package_.first_lidar = false;
      }

      const int32_t lidar_iter_times = lidar_handler_->iter_times();
      const int32_t cam_iter_times = cam_handler_ != nullptr ? cam_handler_->iter_times() : 0;
      const int32_t optic_layers = cam_handler_ != nullptr ? cam_handler_->optical_layers() : 0;
      Eigen::MatrixXf h_lidar, h_cam;  // hessian: n*6 (rot pos)
      Eigen::VectorXf r_lidar, r_cam;
      Eigen::VectorXf c_lidar, c_cam;

      predict_x_ = x_;
      predict_cov_ = cov_;
      for (int32_t liter = 0; liter < lidar_iter_times; ++liter) {
        if (lidar_handler_->Observation(x_, true, &h_lidar, &r_lidar, &c_lidar)) {
          const auto& t1 = omp_get_wtime();
          IeskfUpdate(h_lidar, r_lidar, c_lidar);
          const auto& t2 = omp_get_wtime();
          t_ieskf += (t2 - t1) * 1000.;
        }
      }

      const auto& t_lid_map_inc_begin = omp_get_wtime();
      HybridCloud::Ptr hybrid_cloud = lidar_handler_->MapIncremental(x_);
      const auto& t_lid_map_inc_end = omp_get_wtime();
      t_map_inc += (t_lid_map_inc_end - t_lid_map_inc_begin) * 1000.;

      predict_x_ = x_;
      predict_cov_ = cov_;
      for (int32_t layer = optic_layers - 1; layer >= 0; --layer) {
        for (int32_t citer = 0; citer < cam_iter_times; ++citer) {
          if (cam_handler_ != nullptr &&
              cam_handler_->Observation(x_, layer, &h_cam, &r_cam, &c_cam)) {
            const auto& t1 = omp_get_wtime();
            IeskfUpdate(h_cam, r_cam, c_cam);
            const auto& t2 = omp_get_wtime();
            t_ieskf += (t2 - t1) * 1000.;
          }
        }
      }

      const auto& t_cam_map_inc_begin = omp_get_wtime();
      if (cam_handler_ != nullptr) cam_handler_->RenderScan(x_, hybrid_cloud);
      map_->InsertPoints(hybrid_cloud);
      if (cam_handler_ != nullptr) {
        VoxelVisualPtr vis_cloud = cam_handler_->MapIncremental(x_, hybrid_cloud);
        map_->InsertVisFeats(vis_cloud);
      }
      const auto& t_cam_map_inc_end = omp_get_wtime();
      t_map_inc += (t_cam_map_inc_end - t_cam_map_inc_begin) * 1000.;

      sync_package_.imu_states.clear();
      imu_handler_->InsertImuState(cur_imu_msg, x_, cov_, update_time,
                                   &sync_package_.imu_states);

      lidar_handler_->Publish();
      if (cam_handler_) cam_handler_->Publish();
      PublishPath();
      PublishOdometry();
      PublishHybridMap();

      update_time = std::numeric_limits<double>::infinity();
    }
  }
  sync_package_.clear();

  return true;
}

bool LIVO::IeskfUpdate(const Eigen::MatrixXf& h, const Eigen::VectorXf& r,
                       const Eigen::VectorXf& c) {
  cov_ = predict_cov_;
  Eigen::Matrix<float, Ikfom::kStateDof, 1> dx = x_ - predict_x_;
  // compute rot jacobian error-state-iter-i+i / error-state-iter-i
  // const Eigen::Vector3f& delta_rot = dx.block<3, 1>(0, 0);
  // const Eigen::Matrix3f& jacobian_rot = lie_algebra::RightJacobian(delta_rot);
  // dx.block<3, 1>(0, 0) = jacobian_rot * dx.block<3, 1>(0, 0);
  // cov_.block<3, Ikfom::kStateDof>(0, 0) =
  //     jacobian_rot * cov_.block<3, Ikfom::kStateDof>(0, 0);
  // cov_.block<Ikfom::kStateDof, 3>(0, 0) =
  //     cov_.block<Ikfom::kStateDof, 3>(0, 0) * jacobian_rot.transpose();
  // compute grav jacobian error-state-iter-i+i / error-state-iter-i
  // const Eigen::Vector2f& delta_grav = dx.block<2, 1>(15, 0);
  // const Eigen::Matrix2f& jacobian_grav = x_.grav.S2_Nx_yy() *
  //                                        predict_x_.grav.S2_Mx(delta_grav);
  // dx.block<2, 1>(15, 0) = jacobian_grav * dx.block<2, 1>(15, 0);
  // cov_.block<2, Ikfom::kStateDof>(15, 0) =
  //     jacobian_grav * cov_.block<2, Ikfom::kStateDof>(15, 0);
  // cov_.block<Ikfom::kStateDof, 2>(0, 15) =
  //     cov_.block<Ikfom::kStateDof, 2>(0, 15) * jacobian_grav.transpose();
  // compute kalman gain
  Eigen::MatrixXf htRinv = h.transpose();
  Ikfom::IkfomCovMat hth = Ikfom::IkfomCovMat::Zero();
  for (int32_t col_idx = 0; col_idx < htRinv.cols(); ++col_idx)
    htRinv.col(col_idx) /= c.row(col_idx)(0);
  hth.block<6, 6>(0, 0) += htRinv * h;
  const Ikfom::IkfomCovMat& P_tmp = cov_.inverse() + hth;
  const Ikfom::IkfomCovMat& P_tmp_inv = P_tmp.inverse();
  Eigen::Matrix<float, Ikfom::kStateDof, 1> K_x;
  K_x.setZero();
  K_x += P_tmp_inv.block<Ikfom::kStateDof, 6>(0, 0) * htRinv * r;
  const Ikfom::IkfomCovMat& K_H = P_tmp_inv * hth;
  const Eigen::Matrix<float, Ikfom::kStateDof, 1>& dx_new =
      K_x + (K_H - Ikfom::IkfomCovMat::Identity()) * dx;
  x_ += dx_new;
  cov_ = (Ikfom::IkfomCovMat::Identity() - K_H) * cov_;
  Eigen::Vector3f drot = dx_new.block<3, 1>(0, 0);
  Eigen::Vector3f dtrans = dx_new.block<3, 1>(3, 0);
  if ((drot.norm() * 57.3 < 0.01) && (dtrans.norm() * 100 < 0.015)) {
    return true;
  }
  return false;
}

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
