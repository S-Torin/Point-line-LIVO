/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file s2.cc
 **/

#include "s2.h"

#include <iostream>

#include "so3.h"
#include "common/mtk/lie_algebra.h"

namespace livo {
namespace mtk {
namespace {
const double kMathEpsilon = 1.e-12;
}  // namespace

S2::S2(const Eigen::Vector3f& i_vec) : vec_(i_vec) {}

S2::S2(const S2& other) : vec_(other.vec_) {}

S2& S2::operator=(const S2& other) {
  vec_ = other.vec_;
  return *this;
}

S2 S2::operator+(const Eigen::Vector2f& delta) const {
  const Eigen::Matrix<float, 3, 2>& basis = BasisMatrix();
  const Eigen::Vector3f& rot_vec = basis * delta;
  return S2(SO3::exp(rot_vec).rot() * vec_);
}

S2& S2::operator+=(const Eigen::Vector2f& delta) {
  const Eigen::Matrix<float, 3, 2>& basis = BasisMatrix();
  const Eigen::Vector3f& rot_vec = basis * delta;
  vec_ = SO3::exp(rot_vec).rot() * vec_;
  return *this;
}

S2 S2::operator+(const Eigen::Vector3f& delta) const {
  return S2(SO3::exp(delta).rot() * vec_);
}

S2& S2::operator+=(const Eigen::Vector3f& delta) {
  vec_ = SO3::exp(delta).rot() * vec_;
  return *this;
}

Eigen::Vector2f S2::operator-(const S2& other) const {
  Eigen::Vector2f ret = Eigen::Vector2f::Zero();
  const Eigen::Vector3f& cross_product = SO3::hat(other.vec_) * vec_;
  const double cross_product_norm = cross_product.norm();
  const double inner_product = other.vec_.transpose() * vec_;
  const double theta = atan2(cross_product_norm, inner_product);
  if (fabs(cross_product_norm) < kMathEpsilon) {
    if (inner_product > kMathEpsilon) {  // 0
      ret(0) = 0.;
    } else {  // pi
      ret(0) = M_PI;
    }
  } else {
    const Eigen::Vector3f& rot_norm_vec = cross_product / cross_product_norm;
    const Eigen::Vector2f& manifold_norm_vec = other.BasisMatrix().transpose() *
                                               rot_norm_vec;
    ret = manifold_norm_vec * theta;
  }
  return ret;
}

Eigen::Matrix<float, 3, 2> S2::BasisMatrix() const {
  Eigen::Matrix<float, 3, 2> ret = Eigen::Matrix<float, 3, 2>::Zero();
  const double length = vec_.norm();
  const double denominator1 = length - vec_(2);
  if (denominator1 < kMathEpsilon) {
    ret(0, 0) = -1.;
    ret(1, 1) = 1.;
  } else {
    const double denominator1_inv = 1. / denominator1;
    ret << length - vec_(0) * vec_(0) * denominator1_inv, -vec_(0) * vec_(1) * denominator1_inv,
        -vec_(0) * vec_(1) * denominator1_inv, length - vec_(1) * vec_(1) * denominator1_inv,
        vec_(0), vec_(1);
    ret /= length;
  }
  return ret;
}

Eigen::Matrix<float, 3, 2> S2::S2_Mx(const Eigen::Vector2f& delta) const {
  // grav + error_g partial error_g
  const Eigen::Matrix<float, 3, 2>& basis = BasisMatrix();
  if (delta.norm() < kMathEpsilon) {
    return -SO3::hat(vec_) * basis;
  } else {
    const Eigen::Vector3f& rot_axis = basis * delta;
    return -SO3::exp(rot_axis).matrix() * SO3::hat(vec_) *
           lie_algebra::RightJacobian(rot_axis) * basis;
  }
}

Eigen::Matrix<float, 2, 3> S2::S2_Nx_yy() const {
  // grav - other partial grav
  const double length = vec_.norm();
  assert(length > kMathEpsilon);
  const double length_inv = 1. / length;
  return length_inv * length_inv * BasisMatrix().transpose() * SO3::hat(vec_);
}
}  // namespace mtk
}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
