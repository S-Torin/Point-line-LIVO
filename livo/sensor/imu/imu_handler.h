/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file imu_handler.h
 **/

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "Eigen/Core"
#include "sensor_msgs/Imu.h"
#include "yaml-cpp/yaml.h"

#include "common/ikfom.h"

namespace livo {

class ImuHandler final {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit ImuHandler(const YAML::Node& imu_handler_config);
  ~ImuHandler() = default;
  bool ImuInit(const sensor_msgs::Imu::ConstPtr& imu_msg_in,
               Ikfom::IkfomState* const state,
               Ikfom::IkfomCovMat* const state_cov);
  bool ImuPredict(double cur_sensor_timestamp,
                  bool cur_sensor_is_imu,
                  const sensor_msgs::Imu::ConstPtr& imu_msg_in,
                  Ikfom::IkfomState* const state,
                  Ikfom::IkfomCovMat* const state_cov,
                  std::deque<ImuState>* const imustate_vec);
  void InsertImuState(
      const sensor_msgs::Imu::ConstPtr& imu_msg_in,
      const Ikfom::IkfomState& state,
      const Ikfom::IkfomCovMat& state_cov,
      double insert_timestamp,
      std::deque<ImuState>* const imustate_vec) const;

 private:
  struct livoImuMsg {
    Eigen::Vector3f acc = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyr = Eigen::Vector3f::Zero();
    double timestamp = -1.;  // sec
    livoImuMsg() = default;
    livoImuMsg(const Eigen::Vector3f& i_acc, const Eigen::Vector3f& i_gyr,
               double i_timestamp)
        : acc(i_acc), gyr(i_gyr), timestamp(i_timestamp) {}
    livoImuMsg(const livoImuMsg& other)
        : acc(other.acc), gyr(other.gyr), timestamp(other.timestamp) {}
  };

 private:
  bool AlignToGravity(const Eigen::Vector3f& gravity_tmp,
                      const Eigen::Vector3f& gravity,
                      Eigen::Matrix3f* const rot) const;
  void ComputeAvrImu(const sensor_msgs::Imu::ConstPtr& imu_msg_in,
                     Eigen::Vector3f* const acc_avr,
                     Eigen::Vector3f* const gyr_avr) const;

  int32_t init_iter_times_ = 20;
  bool imu_initialized_ = false;
  bool inverse_acc_direction_ = true;
  float G_ = 9.81;
  Eigen::Vector3f cov_acc_ = Eigen::Vector3f::Ones();
  Eigen::Vector3f cov_gyr_ = Eigen::Vector3f::Ones();
  Eigen::Vector3f cov_gyr_bias_ = Eigen::Vector3f::Ones();
  Eigen::Vector3f cov_acc_bias_ = Eigen::Vector3f::Ones();
  Eigen::Vector3f gravity_ = Eigen::Vector3f(0, 0, -G_);
  Eigen::Vector3f mean_acc_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f mean_gyr_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f last_acc_avr_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f last_gyr_avr_ = Eigen::Vector3f::Zero();
  Eigen::Matrix<float, 12, 12> noise_cov_mat_ =
      Eigen::Matrix<float, 12, 12>::Identity();
  livoImuMsg last_imu_;
  Ikfom::IkfomInput last_in_;
  double last_other_sensor_time_ = -1.;
};

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
