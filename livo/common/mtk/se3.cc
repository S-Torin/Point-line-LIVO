/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file se3.cc
 **/

#include "se3.h"

#include "so3.h"

namespace livo {
namespace se3 {
SE3::SE3(const Eigen::Matrix3f& rot, const Eigen::Vector3f trans)
    : rot_(Eigen::Quaternionf(rot)),
      trans_(trans) {}

SE3::SE3(const Eigen::Quaternionf& rot, const Eigen::Vector3f trans)
    : rot_(rot),
      trans_(trans) {}

SE3::SE3(const Eigen::Matrix4f& transformation)
    : rot_(transformation.block<3, 3>(0, 0)),
      trans_(transformation.block<3, 1>(0, 3)) {}

SE3::SE3(const SE3& other) : rot_(other.rot_),
                             trans_(other.trans_) {}

SE3& SE3::operator=(const SE3& other) {
  rot_ = other.rot_;
  trans_ = other.trans_;
  return *this;
}

SE3 SE3::operator*(const SE3& other) const {
  SE3 result(*this);
  result.trans_ += rot_ * other.trans_;
  result.rot_ *= other.rot_;
  return result;
}

SE3& SE3::operator*=(const SE3& other) {
  trans_ += rot_ * other.trans_;
  rot_ *= other.rot_;
  return *this;
}

Eigen::Vector3f SE3::operator*(const Eigen::Vector3f& xyz) const {
  return rot_ * xyz + trans_;
}

SE3 SE3::inverse() const {
  SE3 inv;
  inv.rot_ = rot_.conjugate();
  inv.trans_ = inv.rot_ * (trans_ * -1.);
  return inv;
}

const SE3 SE3::Identity() {
  SE3 identity;
  identity.rot_.matrix() = Eigen::Matrix3f::Identity();
  identity.trans_ = Eigen::Vector3f::Zero();
  return identity;
}

Eigen::Vector3f SE3::translation() const {
  return trans_;
}

Eigen::Matrix3f SE3::rotation_matrix() const {
  return rot_.matrix();
}

Eigen::Quaternionf SE3::rot() const {
  return rot_;
}

Eigen::Matrix4f SE3::matrix() const {
  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  transformation.block<3, 3>(0, 0) = rot_.matrix();
  transformation.block<3, 1>(0, 3) = trans_;
  return transformation;
}

SE3 SE3::exp(const Eigen::Matrix<float, 6, 1>& delta) {
  Eigen::Vector3f trans = delta.head<3>();
  Eigen::Vector3f theta = delta.tail<3>();
  double theta_norm = theta.norm();
  Eigen::Matrix3f R = mtk::SO3::exp(theta).matrix();
  Eigen::Matrix3f W = mtk::SO3::hat(theta);
  Eigen::Matrix3f W2 = W * W;
  Eigen::Matrix3f V;
  if (theta_norm < 1e-5) {
    V = Eigen::Matrix3f::Identity() + 0.5 * W + (1.0 / 6.0) * W2;
  } else {
    V = Eigen::Matrix3f::Identity() +
        (1.0 - std::cos(theta_norm)) / (theta_norm * theta_norm) * W +
        (theta_norm - std::sin(theta_norm)) / (theta_norm * theta_norm * theta_norm) * W2;
  }
  Eigen::Vector3f t = V * trans;
  return SE3(R, t);
}

}  // namespace se3
}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
