/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file imu_predict_func.h
 **/

#pragma once

#include "Eigen/Core"

#include "common/ikfom.h"

namespace livo {
namespace imu_predict_func {
using MatJacNoise = Eigen::Matrix<float, Ikfom::kStateDof, Ikfom::kNoiseDim>;
Eigen::Matrix<float, Ikfom::kStateDim, 1> KinematicFunc(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input);
Eigen::Matrix<float, Ikfom::kStateDim, 1> KinematicFunc(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input, double dt);
Eigen::Matrix<float, Ikfom::kStateDim, Ikfom::kStateDof> KinematicFuncFx(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input);
Eigen::Matrix<float, Ikfom::kStateDim, Ikfom::kStateDof> KinematicFuncFx(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input, double dt);
Eigen::Matrix<float, Ikfom::kStateDim, Ikfom::kNoiseDim> KinematicFuncFw(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input);
Eigen::Matrix<float, Ikfom::kStateDim, Ikfom::kNoiseDim> KinematicFuncFw(
    const Ikfom::IkfomState& state, const Ikfom::IkfomInput& input, double dt);
void FxFw(const Ikfom::IkfomState& state,
          const Ikfom::IkfomInput& input,
          double dt,
          Ikfom::IkfomCovMat* const jacobian_x,
          MatJacNoise* const jacobian_w);
void ImuPredict(const Ikfom::IkfomInput& input,
                const Ikfom::ImuCovMat& imu_noise_cov,
                double dt,
                Ikfom::IkfomState* const state,
                Ikfom::IkfomCovMat* const state_cov);
}  // namespace imu_predict_func

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
