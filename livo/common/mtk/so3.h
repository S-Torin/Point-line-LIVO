/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file so3.h
 **/

#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace livo {
namespace mtk {
class SO3 {
 public:
  explicit SO3(const Eigen::Quaternionf& i_rot = Eigen::Quaternionf::Identity());
  explicit SO3(const Eigen::Matrix3f& i_rot = Eigen::Matrix3f::Identity());
  SO3(const SO3& other);
  SO3& operator=(const SO3& other);
  SO3 operator*(const SO3& other) const;
  SO3& operator*=(const SO3& other);
  SO3 operator+(const Eigen::Vector3f& delta) const;
  SO3& operator+=(const Eigen::Vector3f& delta);
  Eigen::Vector3f operator-(const SO3& other) const;
  Eigen::Matrix3f matrix() const { return rot_.toRotationMatrix(); }
  Eigen::Quaternionf rot() const { return rot_; }
  Eigen::Vector3f log() const { return log(*this); };
  SO3 inverse() const { return SO3(rot_.conjugate()); }

  static SO3 exp(const Eigen::Vector3f& delta);
  static SO3 exp(const Eigen::Vector3f& delta, double scale);
  static Eigen::Vector3f log(const SO3& other);
  static Eigen::Matrix3f hat(const Eigen::Vector3f& delta);
  static Eigen::Vector3f vee(const Eigen::Matrix3f& delta);

 public:
  const int32_t dof_ = 3;
  const int32_t dim_ = 3;

 private:
  Eigen::Quaternionf rot_ = Eigen::Quaternionf::Identity();
};
}  // namespace mtk

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
