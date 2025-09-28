/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file cam_handler.h
 **/
#pragma once

#include <memory>
#include <vector>
#include <deque>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <yaml-cpp/yaml.h>

#include "common/ikfom.h"
#include "common/mtk/se3.h"
#include "edge_drawing/edge_drawing.hpp"
#include "map/hybrid_map.h"
#include "point_type.h"

namespace livo {
class CamHandler final {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit CamHandler(const YAML::Node& config, ros::NodeHandle& nh,
                      std::shared_ptr<HybridMap> local_map);
  ~CamHandler();

  void InputImage(const sensor_msgs::Image::ConstPtr& img_msg,
                  const LIVOCloud::ConstPtr& scan_world,
                  const Ikfom::IkfomState& x);
  bool Observation(const Ikfom::IkfomState& x, int32_t layer,
                   Eigen::MatrixXf* const h, Eigen::VectorXf* const r,
                   Eigen::VectorXf* const c);
  VoxelVisualPtr MapIncremental(const Ikfom::IkfomState& x,
                                const HybridCloud::Ptr& hybrid_scan);
  void RenderScan(const Ikfom::IkfomState& x,
                  const HybridCloud::Ptr& hybrid_scan);
  void Publish();
  void SavePcd();

  Eigen::Vector3f world2cam(const Eigen::Vector3f& pt_w) const;
  Eigen::Vector3f cam2world(const Eigen::Vector3f& pt_c) const;
  Eigen::Vector2f cam2pixel(const Eigen::Vector3f& pt_c) const;
  int32_t optical_layers() const { return optical_layer_; }
  int32_t iter_times() const { return max_iter_times_; }
  int32_t width() const { return width_; }
  int32_t height() const { return height_; }
  float fx() const { return cam_fx_; }
  float fy() const { return cam_fy_; }
  float cx() const { return cam_cx_; }
  float cy() const { return cam_cy_; }
  double meas_cov() const { return meas_cov_; }
  Eigen::Vector3f pos() const { return tf_wc_.translation(); }
  Eigen::Quaternionf rot() const { return tf_wc_.rot(); }
  Eigen::Matrix3f ric() const { return ric_; }
  Eigen::Vector3f tic() const { return tic_; }
  const cv::Mat& img_rgb() const { return img_rgb_; }
  const cv::Mat& img_gray() const { return img_gray_; }
  const cv::Mat& img_grad() const { return img_grad_; }
  const cv::Mat& img_depth() const { return img_depth_; }
  se3::SE3 pose(const Ikfom::IkfomState& x);
  void update_pose(Ikfom::IkfomState& x, const se3::SE3& pose);
  Eigen::Vector3f get_rgb(Eigen::Vector3f point) const;

 private:
  static float shiTomasiScore(const cv::Mat& img, const Eigen::Vector2i& uv);

  int32_t grid_index(const Eigen::Vector2f& uv) const;
  bool InFov(const Eigen::Vector2f& uv, int32_t boundry) const;
  void set_tf(const Ikfom::IkfomState& x);
  Eigen::Vector3f get_bgr(const Eigen::Vector2f& uv) const;
  float InterpolateMat8u(const cv::Mat& img, const Eigen::Vector2f& uv,
                         Eigen::Matrix<float, 1, 2>* const jacobian = nullptr) const;
  Eigen::Matrix<float, 1, 6> jacobian(const Eigen::Matrix<float, 1, 2>& d_img_uv,
                                      const Eigen::Vector3f& pt_c,
                                      const Ikfom::IkfomState& x) const;
  bool PointLMVDE(const cv::Point2f& pix_point, const Eigen::Vector3f& query_w,
                  VisualFeat& feat_point) const;
  bool LineLMVDE(const cv::Vec4f& pix_line, VisualFeat& feat_line,
                 const std::vector<Eigen::Vector3f>& querys_w,
                 std::vector<cv::Point2f>& pix_samples,
                 std::vector<VisualFeat>& feat_samples) const;
  void get_pfeatpatch(const Eigen::Vector2f& uv, FeatPatch& patch,
                      const Eigen::Matrix2f& affine = Eigen::Matrix2f::Identity()) const;
  void get_lfeatpatch(const Eigen::Vector2f& uv, FeatPatch& patch,
                      const Eigen::Matrix2f& affine = Eigen::Matrix2f::Identity()) const;
  void get_pcolorpatch(const Eigen::Vector2f& uv, ColorPatch& patch,
                       const Eigen::Matrix2f& affine = Eigen::Matrix2f::Identity()) const;
  void get_lcolorpatch(const Eigen::Vector2f& uv, ColorPatch& patch,
                       const Eigen::Matrix2f& affine = Eigen::Matrix2f::Identity()) const;

 private:
  bool enable_ = true;
  bool pcd_save_en_ = false;
  std::string cam_model_ = "Pinhole";
  int32_t width_ = 0, height_ = 0, orig_width_ = 0, orig_height_ = 0;
  float cam_fx_ = 0, cam_fy_ = 0, cam_cx_ = 0, cam_cy_ = 0;
  cv::Mat distort_coeff_;
  cv::Mat K_;
  cv::Rect crop_roi_;
  int32_t grid_nums_ = 0;
  int32_t grid_size_ = 40;
  int32_t half_grid_size_ = 20;
  int32_t optical_layer_ = 1;
  int32_t half_patch_size_ = 4;
  int32_t max_iter_times_ = 3;
  int32_t pub_interval_ = 10;
  int32_t line_sample_num_ = 5;
  double meas_cov_ = 100;
  Eigen::Matrix3f ric_ = Eigen::Matrix3f::Identity();
  Eigen::Vector3f tic_ = Eigen::Vector3f::Zero();

  std::shared_ptr<HybridMap> local_map_ = nullptr;

  std::shared_ptr<cv::ximgproc::EdgeDrawing> ed_ = nullptr;

  double cur_cam_time_ = 0.;
  std_msgs::Header cur_cam_header_;
  se3::SE3 tf_wc_ = se3::SE3::Identity();
  cv::Mat img_rgb_, img_gray_, img_grad_, img_depth_;
  std::vector<cv::Mat> img_pyr_;
  HybridCloud::ConstPtr scan_hybrid_{new HybridCloud};
  ColorCloud::Ptr scan_color_{new ColorCloud};
  ColorCloud::Ptr scan_save_{new ColorCloud};

  std::vector<VoxelVisualPtr> grid_pfeat_;
  std::vector<int32_t> pfeat_indice_;
  std::vector<bool> grid_ptracked_;
  std::vector<bool> grid_pupdate_;
  std::vector<float> grid_pscore_;
  std::vector<float> grid_pdepth_;

  std::vector<VoxelVisualPtr> grid_lfeat_;
  std::vector<int32_t> lfeat_indice_;
  std::vector<bool> grid_ltracked_;
  std::vector<bool> grid_lupdate_;
  std::vector<float> grid_lscore_;
  std::vector<float> grid_ldepth_;

  VoxelVisualPtr vis_feats_{new VoxelVisual};

  std::vector<cv::Point2f> pix_points_;
  std::vector<VisualFeat> feat_points_;

  std::vector<cv::Vec4f> pix_lines_;
  std::vector<VisualFeat> feat_lines_;

  std::vector<cv::Point2f> pix_samples_;
  std::vector<VisualFeat> feat_samples_;

  ros::Publisher pub_feat_;
  ros::Publisher pub_patch_;
  ros::Publisher pub_rgbi_;
  ros::Publisher pub_grad_;
  ros::Publisher pub_points_;
  ros::Publisher pub_lines_;
  ros::Publisher pub_cloud_;
};

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
