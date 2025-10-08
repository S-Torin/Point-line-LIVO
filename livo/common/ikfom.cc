/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file ikfom.cc
 **/

#include "ikfom.h"

namespace livo {
namespace Ikfom {
IkfomState& IkfomState::operator+=(const Eigen::Matrix<float, kStateDof, 1>& other) {
  rot += other.block<3, 1>(0, 0);
  pos += other.block<3, 1>(3, 0);
  vel += other.block<3, 1>(6, 0);
  bg += other.block<3, 1>(9, 0);
  ba += other.block<3, 1>(12, 0);
  grav += static_cast<Eigen::Vector2f>(other.block<2, 1>(15, 0));
  offset_ril += other.block<3, 1>(17, 0);
  offset_til += other.block<3, 1>(20, 0);
  return *this;
}

IkfomState& IkfomState::oplus(const Eigen::Matrix<float, kStateDim, 1>& other, double dt) {
  rot *= mtk::SO3::exp(other.block<3, 1>(0, 0), dt);
  pos += other.block<3, 1>(3, 0) * dt;
  vel += other.block<3, 1>(6, 0) * dt;
  bg += other.block<3, 1>(9, 0) * dt;
  ba += other.block<3, 1>(12, 0) * dt;
  grav.set_vec(grav.vec() + other.block<3, 1>(15, 0) * dt);
  offset_ril *= mtk::SO3::exp(other.block<3, 1>(18, 0), dt);
  offset_til += other.block<3, 1>(21, 0);
  return *this;
}

IkfomState& IkfomState::operator=(const IkfomState& other) {
  rot = other.rot;
  pos = other.pos;
  vel = other.vel;
  bg = other.bg;
  ba = other.ba;
  grav = other.grav;
  offset_ril = other.offset_ril;
  offset_til = other.offset_til;
  return *this;
}

Eigen::Matrix<float, kStateDof, 1> IkfomState::operator-(const IkfomState& other) const {
  Eigen::Matrix<float, kStateDof, 1> ret;
  ret.setZero();
  ret.block<3, 1>(0, 0) = rot - other.rot;
  ret.block<3, 1>(3, 0) = pos - other.pos;
  ret.block<3, 1>(6, 0) = vel - other.vel;
  ret.block<3, 1>(9, 0) = bg - other.bg;
  ret.block<3, 1>(12, 0) = ba - other.ba;
  ret.block<2, 1>(15, 0) = grav - other.grav;
  ret.block<3, 1>(17, 0) = offset_ril - other.offset_ril;
  ret.block<3, 1>(20, 0) = offset_til - other.offset_til;
  return ret;
}

std::ostream& operator<<(std::ostream& os, const IkfomState& obj) {
  os << std::fixed << obj.rot.rot().coeffs().transpose() << ' '
     << obj.pos.transpose() << ' ' << obj.vel.transpose() << ' '
     << obj.bg.transpose() << ' ' << obj.ba.transpose() << ' '
     << obj.grav.vec().transpose() << ' '
     << obj.offset_ril.rot().coeffs().transpose() << ' '
     << obj.offset_til.transpose() << std::endl;
  return os;
}
}  // namespace Ikfom
}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
