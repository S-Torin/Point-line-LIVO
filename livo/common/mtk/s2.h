/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file s2.h
 **/

#pragma once

#include "Eigen/Core"

namespace livo {
namespace mtk {
class S2 {
 public:
  explicit S2(const Eigen::Vector3f& i_vec = Eigen::Vector3f(1.0, 0., 0.));
  S2(const S2& other);
  S2& operator=(const S2& other);
  S2 operator+(const Eigen::Vector2f& delta) const;
  S2& operator+=(const Eigen::Vector2f& delta);
  S2 operator+(const Eigen::Vector3f& other) const;
  S2& operator+=(const Eigen::Vector3f& delta);
  Eigen::Vector2f operator-(const S2& other) const;
  Eigen::Matrix<float, 3, 2> BasisMatrix() const;
  Eigen::Matrix<float, 3, 2> S2_Mx(const Eigen::Vector2f& delta) const;
  Eigen::Matrix<float, 2, 3> S2_Nx_yy() const;
  Eigen::Vector3f vec() const { return vec_; }
  void set_vec(const Eigen::Vector3f& i_vec) { vec_ = i_vec; }

 public:
  const int32_t dof_ = 2;
  const int32_t dim_ = 3;

 private:
  Eigen::Vector3f vec_;
};
}  // namespace mtk

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
