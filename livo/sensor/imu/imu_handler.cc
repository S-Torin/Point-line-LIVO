/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file imu_handler.cc
 **/

#include "imu_handler.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>

#include <glog/logging.h>
#include <omp.h>

#include "imu_predict_func.h"
#include "log.h"

namespace livo {
namespace {
using Eigen::Matrix3f;
using Eigen::Vector3f;
const double kMathEpsilon = 1e-8;
}  // namespace

ImuHandler::ImuHandler(const YAML::Node& config) {
  cov_acc_ = Vector3f::Ones() * config["cov_acc"].as<double>();
  cov_gyr_ = Vector3f::Ones() * config["cov_gyr"].as<double>();
  cov_gyr_bias_ = Vector3f::Ones() * config["cov_gyr_bias"].as<double>();
  cov_acc_bias_ = Vector3f::Ones() * config["cov_acc_bias"].as<double>();
  gravity_.z() = -config["gravity"].as<double>();
  G_ = config["gravity"].as<float>();
  init_iter_times_ = config["init_iter_times"].as<int32_t>();
  inverse_acc_direction_ = config["inverse_acc_direction"].as<bool>();
  LOG(WARNING) << "ImuHandler Init Succ";
}

bool ImuHandler::ImuInit(const sensor_msgs::Imu::ConstPtr& imu_msg_in,
                         Ikfom::IkfomState* const state,
                         Ikfom::IkfomCovMat* const state_cov) {
  if (state == nullptr || state_cov == nullptr) {
    LOG(ERROR) << "[ImuHandler]: ImuInit, null kf_state";
    return false;
  }
  if (imu_initialized_) {
    return true;
  }
  static int32_t iter_times = 0;
  ++iter_times;
  Vector3f cur_acc = {float(imu_msg_in->linear_acceleration.x),
                      float(imu_msg_in->linear_acceleration.y),
                      float(imu_msg_in->linear_acceleration.z)};
  const Vector3f cur_gyr = {float(imu_msg_in->angular_velocity.x),
                            float(imu_msg_in->angular_velocity.y),
                            float(imu_msg_in->angular_velocity.z)};
  if (inverse_acc_direction_) {
    cur_acc *= -1;
  }
  if (iter_times == 1) {
    mean_acc_ = cur_acc;
    mean_gyr_ = cur_gyr;
    return false;
  }
  // static init, accumulate static imu msg
  Vector3f delta_acc = cur_acc - mean_acc_;
  Vector3f delta_gyr = cur_gyr - mean_gyr_;
  mean_acc_ += delta_acc / iter_times;
  mean_gyr_ += delta_gyr / iter_times;
  // init imu state
  if (iter_times > init_iter_times_) {
    Vector3f gravity_tmp = -mean_acc_ / mean_acc_.norm() * G_;
    Matrix3f init_rot = Matrix3f::Identity();
    if (!AlignToGravity(gravity_tmp, gravity_, &init_rot)) {
      LOG(ERROR) << "[ImuHandler]: ImuInit, align gravity fail";
      return false;
    }
    // init kf state
    state->grav = mtk::S2(gravity_tmp);
    state->rot = mtk::SO3(Eigen::Quaternionf::Identity());
    // state->grav = mtk::S2(gravity_);
    // state->rot = mtk::SO3(Eigen::Quaternionf(init_rot));
    state->bg = mean_gyr_;
    // state->ba = mean_acc_ + gravity_tmp;
    // init noise covriance
    noise_cov_mat_.setIdentity();
    noise_cov_mat_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
    noise_cov_mat_.block<3, 3>(3, 3).diagonal() = cov_acc_;
    noise_cov_mat_.block<3, 3>(6, 6).diagonal() = cov_gyr_bias_;
    noise_cov_mat_.block<3, 3>(9, 9).diagonal() = cov_acc_bias_;
    LOG(INFO) << "Imu Init Succ, mean acc:"
              << std::fixed << std::setw(6) << mean_acc_.transpose();
    LOG(INFO) << "Imu Init Succ, mean gyr:"
              << std::fixed << std::setw(6) << mean_gyr_.transpose();
    LOG(INFO) << "Imu Init Succ, g:"
              << std::fixed << std::setw(6) << state->grav.vec().transpose();
    // insert cur img msg
    double cur_imu_timestamp = imu_msg_in->header.stamp.toSec();
    last_imu_ = livoImuMsg(cur_acc, cur_gyr, cur_imu_timestamp);
    imu_initialized_ = true;
    return false;
  }
  LOG(INFO) << "Imu Initializing, mean acc:"
            << std::fixed << std::setw(6) << cur_acc.transpose();
  return false;
}

bool ImuHandler::AlignToGravity(const Vector3f& gravitytmp,
                                const Vector3f& gravity,
                                Matrix3f* const rot) const {
  if (rot == nullptr) {
    LOG(ERROR) << "[ImuHandler]: AlignToGravity, null rot mat";
    return false;
  }
  const double gravitytmp_norm = gravitytmp.norm();
  const double gravity_norm = gravity.norm();
  assert(std::fabs(gravitytmp_norm) > kMathEpsilon &&
         std::fabs(gravity_norm) > kMathEpsilon);
  const double norm_inv = 1. / gravitytmp_norm / gravity_norm;
  const Matrix3f& hat_gravtmp = mtk::SO3::hat(gravitytmp);
  const double align_sin = (hat_gravtmp * gravity).norm() * norm_inv;
  const double align_cos =
      static_cast<double>(gravity.transpose() * gravitytmp) * norm_inv;
  if (align_sin < kMathEpsilon) {
    if (align_cos > kMathEpsilon) {
      *rot = Matrix3f::Identity();
    } else {
      *rot = -Matrix3f::Identity();
    }
  } else {
    Vector3f align_angle = hat_gravtmp * gravity /
                           (hat_gravtmp * gravity).norm() * acos(align_cos);
    *rot = mtk::SO3::exp(align_angle).matrix();
  }
  return true;
}

void ImuHandler::ComputeAvrImu(const sensor_msgs::Imu::ConstPtr& imu_msg_in,
                               Eigen::Vector3f* const acc_avr,
                               Eigen::Vector3f* const gyr_avr) const {
  float acc_inverse = inverse_acc_direction_ ? -1.f : 1.f;
  const Vector3f cur_acc = {acc_inverse * float(imu_msg_in->linear_acceleration.x),
                            acc_inverse * float(imu_msg_in->linear_acceleration.y),
                            acc_inverse * float(imu_msg_in->linear_acceleration.z)};
  const Vector3f cur_gyr = {float(imu_msg_in->angular_velocity.x),
                            float(imu_msg_in->angular_velocity.y),
                            float(imu_msg_in->angular_velocity.z)};
  const float acc_scale = G_ / mean_acc_.norm();
  // const double acc_scale = 1.;
  *acc_avr = 0.5 * (cur_acc + last_imu_.acc) * acc_scale;
  *gyr_avr = 0.5 * (cur_gyr + last_imu_.gyr);
}

void ImuHandler::InsertImuState(
    const sensor_msgs::Imu::ConstPtr& imu_msg_in,
    const Ikfom::IkfomState& state,
    const Ikfom::IkfomCovMat& state_cov,
    double insert_timestamp,
    std::deque<ImuState>* const imustate_vec) const {
  Vector3f acc_avr = Vector3f::Zero();
  Vector3f gyr_avr = Vector3f::Zero();
  ComputeAvrImu(imu_msg_in, &acc_avr, &gyr_avr);
  const Vector3f& acc_w = state.rot.rot() * (acc_avr - state.ba) +
                          state.grav.vec();
  const Vector3f& gyr_w = gyr_avr - state.bg;
  Ikfom::IkfomInput in;
  in.acc = acc_avr;
  in.gyr = gyr_avr;
  imustate_vec->emplace_back(ImuState(insert_timestamp, state.rot.matrix(),
                                      state.pos, state.vel, acc_w, gyr_w, in));
  imustate_vec->back().x = state;
  imustate_vec->back().P = state_cov;
  imustate_vec->back().x_predict = state;
  imustate_vec->back().P_predict = state_cov;
  imustate_vec->back().imu_noise_cov = noise_cov_mat_;
}

bool ImuHandler::ImuPredict(double cur_sensor_timestamp,
                            bool cur_sensor_is_imu,
                            const sensor_msgs::Imu::ConstPtr& imu_msg_in,
                            Ikfom::IkfomState* const state,
                            Ikfom::IkfomCovMat* const state_cov,
                            std::deque<ImuState>* const imustate_vec) {
  const auto& t1 = omp_get_wtime();
  assert(last_imu_.timestamp > 0. && imu_msg_in != nullptr &&
         state != nullptr && state_cov != nullptr && imustate_vec != nullptr);
  if (cur_sensor_is_imu) {
    assert(std::fabs(cur_sensor_timestamp - imu_msg_in->header.stamp.toSec()) <
           kMathEpsilon);
  } else {
    assert(imu_msg_in->header.stamp.toSec() - cur_sensor_timestamp >
           kMathEpsilon);
  }
  float acc_inverse = inverse_acc_direction_ ? -1.f : 1.f;
  const Vector3f cur_acc = {acc_inverse * float(imu_msg_in->linear_acceleration.x),
                            acc_inverse * float(imu_msg_in->linear_acceleration.y),
                            acc_inverse * float(imu_msg_in->linear_acceleration.z)};
  const Vector3f cur_gyr = {float(imu_msg_in->angular_velocity.x),
                            float(imu_msg_in->angular_velocity.y),
                            float(imu_msg_in->angular_velocity.z)};
  const float acc_scale = G_ / mean_acc_.norm();
  // const double acc_scale = 1.;
  const Vector3f& acc_avr = 0.5 * (cur_acc + last_imu_.acc) * acc_scale;
  const Vector3f& gyr_avr = 0.5 * (cur_gyr + last_imu_.gyr);
  // predict
  float dt = cur_sensor_timestamp - std::fmax(last_other_sensor_time_,
                                              last_imu_.timestamp);
  LOG(INFO) << "imu predict dt:" << std::fixed << dt;
  Ikfom::IkfomInput in;
  in.acc = acc_avr;
  in.gyr = gyr_avr;

  if (cur_sensor_is_imu) {
    last_in_ = in;
  } else {
    in = last_in_;
  }

  imu_predict_func::ImuPredict(in, noise_cov_mat_, dt, state, state_cov);
  const Vector3f& acc_w = state->rot.rot() * (acc_avr - state->ba) +
                          state->grav.vec();
  const Vector3f& gyr_w = gyr_avr - state->bg;
  imustate_vec->emplace_back(ImuState(cur_sensor_timestamp,
                                      state->rot.matrix(), state->pos,
                                      state->vel, acc_w, gyr_w, in));
  imustate_vec->back().x = *state;
  imustate_vec->back().P = *state_cov;
  imustate_vec->back().x_predict = *state;
  imustate_vec->back().P_predict = *state_cov;
  imustate_vec->back().imu_noise_cov = noise_cov_mat_;
  if (cur_sensor_is_imu) {
    last_imu_ = livoImuMsg(cur_acc, cur_gyr, cur_sensor_timestamp);
    last_other_sensor_time_ = -1.;
  } else {
    last_other_sensor_time_ = cur_sensor_timestamp;
  }
  const auto& t2 = omp_get_wtime();
  t_imu_pre += (t2 - t1) * 1000.;
  return true;
}

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
