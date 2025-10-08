/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file imu_predict_func.cc
 **/

#include "imu_predict_func.h"

#include <fstream>
#include <iomanip>

#include "glog/logging.h"

#include "common/mtk/so3.h"
#include "common/mtk/lie_algebra.h"

namespace livo {

namespace {
using Eigen::Matrix;
using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Vector2f;
using Eigen::Vector3f;
}  // namespace
namespace imu_predict_func {
Matrix<float, Ikfom::kStateDim, 1> KinematicFunc(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input, double dt) {
  return KinematicFunc(state, input) * dt;
}

Matrix<float, Ikfom::kStateDim, 1> KinematicFunc(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input) {
  Matrix<float, Ikfom::kStateDim, 1> kinefun_vec;
  kinefun_vec.setZero();
  kinefun_vec.block<3, 1>(0, 0) = input.gyr - state.bg;  // d_rot = gyr
  kinefun_vec.block<3, 1>(3, 0) = state.vel;             // d_pos = vel
  kinefun_vec.block<3, 1>(6, 0) = state.rot.rot() * (input.acc - state.ba) +
                                  state.grav.vec();  // d_vel = acc
  return kinefun_vec;
}

Matrix<float, Ikfom::kStateDim, Ikfom::kStateDof> KinematicFuncFx(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input) {
  // d kinematic func d error state i-1
  Matrix<float, Ikfom::kStateDim, Ikfom::kStateDof> jacobian;
  jacobian.setZero();
  jacobian.block<3, 3>(0, 9) = -Matrix3f::Identity();
  jacobian.block<3, 3>(3, 6) = Matrix3f::Identity();
  jacobian.block<3, 3>(6, 0) = -state.rot.matrix() *
                               mtk::SO3::hat(input.acc - state.ba);
  jacobian.block<3, 3>(6, 12) = -state.rot.matrix();
  jacobian.block<3, 2>(6, 15) = state.grav.S2_Mx(Vector2f::Zero());
  return jacobian;
}

Matrix<float, Ikfom::kStateDim, Ikfom::kStateDof> KinematicFuncFx(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input, double dt) {
  return KinematicFuncFx(state, input) * dt;
}

Matrix<float, Ikfom::kStateDim, Ikfom::kNoiseDim> KinematicFuncFw(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input) {
  // d kinematic-func d noise-i-1
  Matrix<float, Ikfom::kStateDim, Ikfom::kNoiseDim> jacobian;
  jacobian.setZero();
  jacobian.block<3, 3>(0, 0) = -Matrix3f::Identity();
  jacobian.block<3, 3>(6, 3) = -state.rot.matrix();
  jacobian.block<3, 3>(9, 6) = Matrix3f::Identity();
  jacobian.block<3, 3>(12, 9) = Matrix3f::Identity();
  return jacobian;
}

Matrix<float, Ikfom::kStateDim, Ikfom::kNoiseDim> KinematicFuncFw(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input, double dt) {
  return KinematicFuncFw(state, input) * dt;
}

void FxFw(const Ikfom::IkfomState& state,
          const Ikfom::IkfomInput& input,
          double dt,
          Ikfom::IkfomCovMat* const jacobian_x,
          MatJacNoise* const jacobian_w) {
  Matrix<float, Ikfom::kStateDim, 1> kinefun_vec =
      KinematicFunc(state, input, dt);
  Matrix<float, Ikfom::kStateDof, Ikfom::kStateDim> jacobian_kinefunc;
  jacobian_kinefunc.setZero();
  jacobian_x->setIdentity();  // jacobian error-state-i-1
  jacobian_w->setZero();      // jacobian error-state-noise-i-1
  // d error-state-i / d error-state-i-1
  const Matrix3f& rot_jacobian_error =
      mtk::SO3::exp(-1. * kinefun_vec.block<3, 1>(0, 0)).matrix();
  const Vector3f& kinefunc_grav = kinefun_vec.block<3, 1>(15, 0);
  const Matrix<float, 2, 3>& Nx = (state.grav + kinefunc_grav).S2_Nx_yy();
  const Matrix<float, 3, 2>& Mx = state.grav.S2_Mx(Eigen::Vector2f::Zero());
  const Matrix2f& grav_jacobian_error =
      Nx * mtk::SO3::exp(kinefunc_grav).matrix() * Mx;
  // d error-state-i / d kinematic-func
  const Matrix3f& rot_jacobian_kinefunc =
      lie_algebra::RightJacobian(-kinefun_vec.block<3, 1>(0, 0));
  const Matrix<float, 2, 3>& grav_jacobian_kinefunc =
      -Nx * mtk::SO3::exp(kinefunc_grav).matrix() *
      mtk::SO3::hat(state.grav.vec()) *
      lie_algebra::RightJacobian(kinefunc_grav);
  // compute jacobian_kinefunc
  jacobian_kinefunc.block<3, 3>(0, 0) = rot_jacobian_kinefunc;     // rot
  jacobian_kinefunc.block<3, 3>(3, 3).setIdentity();               // pos
  jacobian_kinefunc.block<3, 3>(6, 6).setIdentity();               // vel
  jacobian_kinefunc.block<3, 3>(9, 9).setIdentity();               // bg
  jacobian_kinefunc.block<3, 3>(12, 12).setIdentity();             // ba
  jacobian_kinefunc.block<2, 3>(15, 15) = grav_jacobian_kinefunc;  // grav
  // compute jacobian_x
  jacobian_x->block<3, 3>(0, 0) = rot_jacobian_error;     // rot
  jacobian_x->block<2, 2>(15, 15) = grav_jacobian_error;  // grav
  const Matrix<float, Ikfom::kStateDim, Ikfom::kStateDof>
      kinefunc_jacobian_error = KinematicFuncFx(state, input);
  *jacobian_x += jacobian_kinefunc * kinefunc_jacobian_error * dt;
  // compute jacobian_w
  const Matrix<float, Ikfom::kStateDim, Ikfom::kNoiseDim>&
      kinefunc_jacobian_noise = KinematicFuncFw(state, input);
  *jacobian_w += jacobian_kinefunc * kinefunc_jacobian_noise * dt;
}

void ImuPredict(const Ikfom::IkfomInput& input,
                const Ikfom::ImuCovMat& imu_noise_cov,
                double dt,
                Ikfom::IkfomState* const state,
                Ikfom::IkfomCovMat* const state_cov) {
  Matrix<float, Ikfom::kStateDof, Ikfom::kStateDof> jacobian_x;
  Matrix<float, Ikfom::kStateDof, Ikfom::kNoiseDim> jacobian_w;
  FxFw(*state, input, dt, &jacobian_x, &jacobian_w);
  Matrix<float, Ikfom::kStateDim, 1> kinefun_vec = KinematicFunc(*state, input);
  // *state += kinefun_vec;
  state->oplus(kinefun_vec, dt);
  *state_cov = jacobian_x * *state_cov * jacobian_x.transpose() +
               jacobian_w * imu_noise_cov * jacobian_w.transpose();
}
}  // namespace imu_predict_func
}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
