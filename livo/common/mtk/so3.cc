/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file so3.cc
 **/

#include "so3.h"

namespace livo {
namespace mtk {
namespace {
const double kMathEpsilon = 1.e-12;
}  // namespace

SO3::SO3(const Eigen::Quaternionf& i_rot) : rot_(i_rot) {}

SO3::SO3(const Eigen::Matrix3f& i_rot) : rot_(Eigen::Quaternionf(i_rot)) {}

SO3::SO3(const SO3& other) : rot_(other.rot_) {}

SO3& SO3::operator=(const SO3& other) {
  rot_ = other.rot_;
  return *this;
}

SO3 SO3::operator*(const SO3& other) const {
  SO3 ret(rot_ * other.rot_);
  ret.rot_.normalize();
  return ret;
}

SO3& SO3::operator*=(const SO3& other) {
  rot_ = rot_ * other.rot_;
  rot_.normalize();
  return *this;
}

SO3 SO3::operator+(const Eigen::Vector3f& delta) const {
  SO3 ret(*this * exp(delta));
  ret.rot_.normalize();
  return ret;
}

SO3& SO3::operator+=(const Eigen::Vector3f& delta) {
  *this = *this * exp(delta);
  rot_.normalize();
  return *this;
}

Eigen::Vector3f SO3::operator-(const SO3& other) const {
  return log(other.inverse() * *this);
}

SO3 SO3::exp(const Eigen::Vector3f& omega) {
  double theta = omega.norm();
  if (std::fabs(theta) < kMathEpsilon) {
    return SO3(Eigen::Quaternionf::Identity());
  } else {
    Eigen::Quaternionf q;
    q.w() = std::cos(theta * 0.5);
    q.vec() = std::sin(theta * 0.5) * omega / theta;
    return SO3(q);
  }
}

SO3 SO3::exp(const Eigen::Vector3f& omega, double scale) {
  double theta = omega.norm();
  if (std::fabs(theta) < kMathEpsilon) {
    return SO3(Eigen::Quaternionf::Identity());
  } else {
    Eigen::Quaternionf q;
    q.w() = std::cos(theta * 0.5 * scale);
    q.vec() = std::sin(theta * 0.5 * scale) * omega / theta;
    return SO3(q);
  }
}

Eigen::Vector3f SO3::log(const SO3& other) {
  const double cos_half_theta = other.rot_.w();
  double sin_half_theta = other.rot_.vec().norm();
  if (sin_half_theta < kMathEpsilon) {
    return Eigen::Vector3f::Zero();
  }
  const double half_theta = std::atan(sin_half_theta / cos_half_theta);
  return Eigen::Vector3f(2 * half_theta * other.rot_.vec() / sin_half_theta);
}

Eigen::Matrix3f SO3::hat(const Eigen::Vector3f& delta) {
  Eigen::Matrix3f ret = Eigen::Matrix3f::Identity();
  ret << 0., -delta(2), delta(1),
      delta(2), 0., -delta(0),
      -delta(1), delta(0), 0.;
  return ret;
}

Eigen::Vector3f SO3::vee(const Eigen::Matrix3f& delta) {
  assert(std::fabs(delta(2, 1) + delta(1, 2)) < kMathEpsilon);
  assert(std::fabs(delta(0, 2) + delta(2, 0)) < kMathEpsilon);
  assert(std::fabs(delta(1, 0) + delta(0, 1)) < kMathEpsilon);
  return Eigen::Vector3f(delta(2, 1), delta(0, 2), delta(1, 0));
}

}  // namespace mtk
}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
