/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file lie_algebra.h
 **/

#pragma once

#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace livo {
namespace lie_algebra {
Eigen::Matrix3f LeftJacobian(const Eigen::Vector3f& rot);
Eigen::Matrix3f LeftJacobianInv(const Eigen::Vector3f& rot);
Eigen::Matrix3f RightJacobian(const Eigen::Vector3f& rot);
Eigen::Matrix3f RightJacobianInv(const Eigen::Vector3f& rot);
}  // namespace lie_algebra
}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
