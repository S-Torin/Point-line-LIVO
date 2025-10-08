
/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file voxel_hash.h
 **/

#include <unordered_map>

#pragma once

class VOXEL_KEY {
 public:
  int64_t x, y, z;
  VOXEL_KEY(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}
  bool operator==(const VOXEL_KEY& other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
  static VOXEL_KEY from_hash(int64_t hash_value) {
    int64_t x = (hash_value >> 40) & 0xFFFFFF;
    if (x & 0x800000) x |= ~0xFFFFFF;
    int64_t y = (hash_value >> 16) & 0xFFFFFF;
    if (y & 0x800000) y |= ~0xFFFFFF;
    int64_t z = hash_value & 0xFFFF;
    if (z & 0x8000) z |= ~0xFFFF;
    return VOXEL_KEY(x, y, z);
  }
};

namespace std {
template <>
struct hash<VOXEL_KEY> {
  int64_t operator()(const VOXEL_KEY& s) const {
    int64_t x_shifted = (s.x & 0xFFFFFF) << 40;
    int64_t y_shifted = (s.y & 0xFFFFFF) << 16;
    int64_t z_shifted = (s.z & 0xFFFF);
    return x_shifted | y_shifted | z_shifted;
  }
};
}  // namespace std
