/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file se3.h
 **/

#pragma once

#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace livo {

namespace se3 {
class SE3 final {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit SE3(const Eigen::Matrix3f& rot = Eigen::Matrix3f::Identity(),
               const Eigen::Vector3f trans = Eigen::Vector3f::Zero());
  explicit SE3(const Eigen::Quaternionf& rot,
               const Eigen::Vector3f trans = Eigen::Vector3f::Zero());
  explicit SE3(const Eigen::Matrix4f& transformation);
  SE3(const SE3& other);
  ~SE3() = default;

  SE3& operator=(const SE3& other);
  SE3 operator*(const SE3& other) const;
  SE3& operator*=(const SE3& other);
  Eigen::Vector3f operator*(const Eigen::Vector3f& xyz) const;

  SE3 inverse() const;
  static const SE3 Identity();
  Eigen::Vector3f translation() const;
  Eigen::Matrix3f rotation_matrix() const;
  Eigen::Quaternionf rot() const;
  Eigen::Matrix4f matrix() const;
  static SE3 exp(const Eigen::Matrix<float, 6, 1>& delta);

 private:
  Eigen::Quaternionf rot_ = Eigen::Quaternionf::Identity();
  Eigen::Vector3f trans_ = Eigen::Vector3f::Zero();
};
}  // namespace se3

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
