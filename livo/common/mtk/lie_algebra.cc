/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file lie_algebra.cc
 **/

#include "lie_algebra.h"

#include "common/mtk/so3.h"

namespace livo {
namespace lie_algebra {
namespace {
const double kMathEpsilon = 1.e-12;
using Eigen::Matrix3f;
using Eigen::Vector3f;
}  // namespace

Eigen::Matrix3f LeftJacobian(const Eigen::Vector3f& rot) {
  double theta = rot.norm();
  if (std::fabs(theta) <= kMathEpsilon) {
    return Matrix3f::Identity();
  }
  double theta_inv = 1. / theta;
  const Vector3f& a = rot * theta_inv;
  return (sin(theta) * theta_inv) * Matrix3f::Identity() +
         (1 - sin(theta) * theta_inv) * a * a.transpose() +
         (1 - cos(theta)) * theta_inv * mtk::SO3::hat(a);
}

Eigen::Matrix3f LeftJacobianInv(const Eigen::Vector3f& rot) {
  double theta = rot.norm();
  if (std::fabs(theta) <= kMathEpsilon) {
    return Matrix3f::Identity();
  }
  double half_theta = theta * 0.5;
  double cot_half_theta = 1. / tan(half_theta);
  double theta_inv = 1. / theta;
  const Vector3f& a = rot * theta_inv;
  return half_theta * cot_half_theta * Matrix3f::Identity() +
         (1 - half_theta * cot_half_theta) * a * a.transpose() -
         half_theta * mtk::SO3::hat(a);
}

Eigen::Matrix3f RightJacobian(const Eigen::Vector3f& rot) {
  return LeftJacobian(rot).transpose();
}

Eigen::Matrix3f RightJacobianInv(const Eigen::Vector3f& rot) {
  return LeftJacobianInv(rot).transpose();
}
}  // namespace lie_algebra
}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
