/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file run_livo.cc
 **/

#include <csignal>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ros/ros.h>

#include "livo.h"

bool FLAG_EXIT = false;
void SigHandle(int sig) {
  FLAG_EXIT = true;
  LOG(WARNING) << "catch sig " << sig;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::WARNING);
  FLAGS_colorlogtostderr = true;
  FLAGS_log_dir = ros::param::param<std::string>("/glog_location",
                                                 std::string(ROOT_DIR) + "Log/glog");

  ros::init(argc, argv, "livo", ros::InitOption::NoSigintHandler);
  ros::NodeHandle nh;
  std::string config_path = ros::param::param<std::string>("/config_path", "none");
  YAML::Node config = YAML::LoadFile(config_path);

  std::shared_ptr<livo::LIVO> livo_process(new livo::LIVO(config, nh));

  signal(SIGINT, SigHandle);
  ros::Rate rate(5000);

  while (ros::ok()) {
    if (FLAG_EXIT) {
      break;
    }
    ros::spinOnce();
    livo_process->Process();
    rate.sleep();
  }
  ros::shutdown();
  google::ShutdownGoogleLogging();

  return 0;
}

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
