/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file log.h
 **/

#pragma once

#include <iostream>

namespace livo {
extern int32_t log_pos_index;

extern double t_imu_pre;
extern double t_lid_pre;
extern double t_vis_pre;
extern double t_vis_obs;
extern double t_lid_obs;
extern double t_ieskf;
extern double t_map_inc;
extern double t_total;
}  // namespace livo
