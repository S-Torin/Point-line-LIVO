/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file ikfom.h
 **/

#pragma once

#include <iostream>

#include "Eigen/Core"

#include "common/ikfom.h"
#include "common/mtk/s2.h"
#include "common/mtk/so3.h"

namespace livo {
namespace Ikfom {
constexpr int kStateDim = 24;
constexpr int kStateDof = 23;
struct IkfomState {
  mtk::SO3 rot = mtk::SO3(Eigen::Quaternionf::Identity());
  Eigen::Vector3f pos = Eigen::Vector3f::Zero();
  Eigen::Vector3f vel = Eigen::Vector3f::Zero();
  Eigen::Vector3f bg = Eigen::Vector3f::Zero();
  Eigen::Vector3f ba = Eigen::Vector3f::Zero();
  mtk::S2 grav = mtk::S2(Eigen::Vector3f(0., 0., 9.81));
  // lidar
  mtk::SO3 offset_ril = mtk::SO3(Eigen::Quaternionf::Identity());
  Eigen::Vector3f offset_til = Eigen::Vector3f::Zero();

  IkfomState& oplus(const Eigen::Matrix<float, kStateDim, 1>& other, double dt);
  IkfomState& operator+=(const Eigen::Matrix<float, kStateDof, 1>& other);
  IkfomState& operator=(const IkfomState& other);
  Eigen::Matrix<float, kStateDof, 1> operator-(const IkfomState& other) const;
  friend std::ostream& operator<<(std::ostream& os, const IkfomState& obj);
};
using IkfomCovMat = Eigen::Matrix<float, kStateDof, kStateDof>;

struct IkfomInput {
  Eigen::Vector3f acc = Eigen::Vector3f::Zero();
  Eigen::Vector3f gyr = Eigen::Vector3f::Zero();
};

struct IkfomNoise {
  Eigen::Vector3f ng = Eigen::Vector3f::Zero();
  Eigen::Vector3f na = Eigen::Vector3f::Zero();
  Eigen::Vector3f nbg = Eigen::Vector3f::Zero();
  Eigen::Vector3f nba = Eigen::Vector3f::Zero();
};

constexpr int kNoiseDim = 12;
using ImuCovMat = Eigen::Matrix<float, kNoiseDim, kNoiseDim>;
}  // namespace Ikfom

struct ImuState {
  double timestamp = 0.;  // sec
  Eigen::Matrix3f rot_w = Eigen::Matrix3f::Identity();
  Eigen::Vector3f pos_w = Eigen::Vector3f::Zero();
  Eigen::Vector3f vel_w = Eigen::Vector3f::Zero();
  Eigen::Vector3f acc_w = Eigen::Vector3f::Zero();
  Eigen::Vector3f gyr_w = Eigen::Vector3f::Zero();

  Ikfom::IkfomInput imu;
  Ikfom::ImuCovMat imu_noise_cov;
  Ikfom::IkfomState x;
  Ikfom::IkfomCovMat P;
  Ikfom::IkfomState x_predict;
  Ikfom::IkfomCovMat P_predict;

  ImuState(double i_timestamp, const Eigen::Matrix3f& i_rot_w,
           const Eigen::Vector3f& i_pos_w, const Eigen::Vector3f& i_vel_w,
           const Eigen::Vector3f& i_acc_w, const Eigen::Vector3f& i_gyr_w,
           const Ikfom::IkfomInput& i_imu)
      : timestamp(i_timestamp),
        rot_w(i_rot_w),
        pos_w(i_pos_w),
        vel_w(i_vel_w),
        acc_w(i_acc_w),
        gyr_w(i_gyr_w),
        imu(i_imu) {};
  ImuState& operator=(const ImuState& other) {
    timestamp = other.timestamp;
    rot_w = other.rot_w;
    pos_w = other.pos_w;
    vel_w = other.vel_w;
    acc_w = other.acc_w;
    gyr_w = other.gyr_w;
    imu = other.imu;
    x = other.x;
    P = other.P;
    return *this;
  };
};
}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
