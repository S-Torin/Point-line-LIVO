/****************************************************************************
 *
 * Copyright (c) 2024 shitong_2001@163.com. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file cam_handler.cc
 **/

#include "cam_handler.h"

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <set>
#include <unordered_set>

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <omp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "log.h"
#include "voxel_hash.h"

namespace livo {
namespace {
const double kMathEpsilon = 1.e-12;
}  // namespace

using namespace Eigen;

CamHandler::CamHandler(const YAML::Node& config, ros::NodeHandle& nh,
                       std::shared_ptr<HybridMap> local_map)
    : local_map_(local_map) {
  enable_ = config["enable"].as<bool>(true);
  pcd_save_en_ = config["pcd_save_en"].as<bool>(true);
  cam_model_ = config["cam_model"].as<std::string>();
  assert(cam_model_ == "Pinhole");
  float scale = config["scale"].as<float>();
  width_ = config["cam_width"].as<int32_t>() / scale;
  height_ = config["cam_height"].as<int32_t>() / scale;
  orig_width_ = width_;
  orig_height_ = height_;
  int32_t crop_width = config["crop_width"].as<int32_t>(-1) / scale;
  int32_t crop_height = config["crop_height"].as<int32_t>(-1) / scale;
  std::vector<float> intrinsic = config["intrinsic"].as<std::vector<float>>();
  cam_fx_ = intrinsic[0] / scale;
  cam_fy_ = intrinsic[1] / scale;
  cam_cx_ = intrinsic[2] / scale;
  cam_cy_ = intrinsic[3] / scale;
  if (crop_width > 0) {
    cam_cx_ -= (width_ - crop_width) / 2.;
    width_ = crop_width;
  }
  if (crop_height > 0) {
    cam_cy_ -= (height_ - crop_height) / 2.;
    height_ = crop_height;
  }
  int32_t start_x = (orig_width_ - width_) / 2;
  int32_t start_y = (orig_height_ - height_) / 2;
  crop_roi_ = cv::Rect(start_x, start_y, width_, height_);
  K_ = (cv::Mat_<double>(3, 3) << cam_fx_, 0., cam_cx_, 0., cam_fy_, cam_cy_, 0., 0., 1.);
  std::vector<float> coeff = config["coeff"].as<std::vector<float>>();
  distort_coeff_ = (cv::Mat_<double>(1, 5) << coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]);
  std::vector<float> extrin_ric = config["offset_ric"].as<std::vector<float>>();
  std::vector<float> extrin_tic = config["offset_tic"].as<std::vector<float>>();
  ric_ << extrin_ric[0], extrin_ric[1], extrin_ric[2],
      extrin_ric[3], extrin_ric[4], extrin_ric[5],
      extrin_ric[6], extrin_ric[7], extrin_ric[8];
  tic_ << extrin_tic[0], extrin_tic[1], extrin_tic[2];

  ed_ = cv::ximgproc::createEdgeDrawing();
  ed_->params.MinLineLength = 50;
  ed_->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
  ed_->params.GradientThresholdValue = 20;
  ed_->params.MinLineLength = 100;

  grid_size_ = config["grid_size"].as<int32_t>(40);
  half_grid_size_ = grid_size_ >> 1;
  half_patch_size_ = config["patch_size"].as<int32_t>(4) >> 1;
  optical_layer_ = config["optical_layer"].as<int32_t>(3);
  max_iter_times_ = config["max_iter_times"].as<int32_t>(3);
  pub_interval_ = config["pub_interval"].as<int32_t>(1);
  meas_cov_ = config["meas_cov"].as<double>(100.);
  img_pyr_.resize(optical_layer_);

  grid_nums_ = (width_ / grid_size_ + 1) * (height_ / grid_size_ + 1);
  grid_pfeat_.resize(grid_nums_, nullptr);
  pfeat_indice_.resize(grid_nums_, -1);
  grid_ptracked_.resize(grid_nums_, false);
  grid_pupdate_.resize(grid_nums_, false);
  grid_pscore_.resize(grid_nums_, 0.f);
  grid_pdepth_.resize(grid_nums_, std::numeric_limits<float>::max());
  grid_lfeat_.resize(grid_nums_, nullptr);
  lfeat_indice_.resize(grid_nums_, -1);
  grid_ltracked_.resize(grid_nums_, false);
  grid_lupdate_.resize(grid_nums_, false);
  grid_lscore_.resize(grid_nums_, 0.f);
  grid_ldepth_.resize(grid_nums_, std::numeric_limits<float>::max());

  LOG(INFO) << "[CamHandler]: Init, extrinsic rot_imu_camera: " << std::endl
            << ric_ << std::endl
            << tic_;
  pub_feat_ = nh.advertise<sensor_msgs::Image>("feat_img", 100000);
  pub_patch_ = nh.advertise<sensor_msgs::Image>("patch_img", 100000);
  pub_rgbi_ = nh.advertise<sensor_msgs::Image>("rgbi_img", 100000);
  pub_grad_ = nh.advertise<sensor_msgs::Image>("grad_img", 100000);
  pub_points_ = nh.advertise<visualization_msgs::MarkerArray>("vis_points", 100000);
  pub_lines_ = nh.advertise<visualization_msgs::MarkerArray>("vis_lines", 100000);
  pub_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("cloud_color", 100000);
  LOG(WARNING) << "CamHandler Init Succ";
}

CamHandler::~CamHandler() {
  std::cout << "\033[32mCamHandler Destruction!\033[0m" << std::endl;
}

void CamHandler::InputImage(const sensor_msgs::Image::ConstPtr& img_msg,
                            const LIVOCloud::ConstPtr& scan_world,
                            const Ikfom::IkfomState& x) {
  double t1 = omp_get_wtime();  // preprocess image
  set_tf(x);
  cur_cam_time_ = img_msg->header.stamp.toSec();
  cur_cam_header_ = img_msg->header;
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
  if (cv_ptr->image.rows != orig_height_ || cv_ptr->image.cols != orig_width_) {
    cv::resize(cv_ptr->image, cv_ptr->image, cv::Size(orig_width_, orig_height_));
  }
  cv_ptr->image = cv_ptr->image(crop_roi_);
  cv::undistort(cv_ptr->image, img_rgb_, K_, distort_coeff_);
  cv::cvtColor(img_rgb_, img_gray_, cv::COLOR_BGR2GRAY);
  cv::Mat grad_x, grad_y;
  cv::Sobel(img_gray_, grad_x, CV_32FC1, 1, 0, 1, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(img_gray_, grad_y, CV_32FC1, 0, 1, 1, 1, 0, cv::BORDER_REPLICATE);
  cv::magnitude(grad_x, grad_y, img_grad_);
  // img_gray_.convertTo(img_pyr_.at(0), CV_8UC1);
  img_grad_.convertTo(img_pyr_.at(0), CV_8UC1);

  // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
  // clahe->apply(img_pyr_.at(0), img_pyr_.at(0));

  for (int32_t i = 1; i < optical_layer_; ++i) {
    cv::resize(img_pyr_.at(i - 1), img_pyr_.at(i), cv::Size(), 0.5, 0.5, cv::INTER_AREA);
  }

  double t2 = omp_get_wtime();  // track vis points and lines from map
  grid_pfeat_.assign(grid_nums_, nullptr);
  pfeat_indice_.assign(grid_nums_, -1);
  grid_ptracked_.assign(grid_nums_, false);
  grid_pupdate_.assign(grid_nums_, false);
  grid_pscore_.assign(grid_nums_, 0.f);
  grid_pdepth_.assign(grid_nums_, std::numeric_limits<float>::max());
  grid_lfeat_.assign(grid_nums_, nullptr);
  lfeat_indice_.assign(grid_nums_, -1);
  grid_ltracked_.assign(grid_nums_, false);
  grid_lupdate_.assign(grid_nums_, false);
  grid_lscore_.assign(grid_nums_, 0.f);
  grid_ldepth_.assign(grid_nums_, std::numeric_limits<float>::max());

  std::unordered_set<VOXEL_KEY> voxel_keys;
  for (const auto& point : scan_world->points) {
    VOXEL_KEY key = local_map_->voxel_indice(point.getVector3fMap());
    voxel_keys.insert(key);
  }
  std::vector<VOXEL_KEY> map_keys = local_map_->FlattenKeys();
  for (const auto& key : map_keys) {
    const Eigen::Vector3f center_w = local_map_->voxel_center(key);
    const Eigen::Vector3f center_c = world2cam(center_w);
    if (center_c.z() < kMathEpsilon) continue;
    const Eigen::Vector2f uv = cam2pixel(center_c);
    if (!InFov(uv, 1)) continue;
    voxel_keys.insert(key);
  }

  std::vector<VOXEL_KEY> keys_vec(voxel_keys.begin(), voxel_keys.end());
  std::vector<VoxelVisualPtr> map_feats;
  if (!local_map_->GetVoxelVisFeats(keys_vec, map_feats)) return;
  for (const auto& feats : map_feats) {
    for (int32_t i = 0; i < feats->feats.size(); ++i) {
      if (!feats->feats.at(i).is_valid) continue;
      const Vector3f& pt_w = feats->feats.at(i).getVector3fMap();
      const Vector3f& pt_c = world2cam(pt_w);
      if (pt_c.z() < kMathEpsilon) continue;
      const Vector3f du_w = {feats->feats.at(i).du_x, feats->feats.at(i).du_y,
                             feats->feats.at(i).du_z};
      const Vector3f du_c = world2cam(du_w);
      if (du_c.z() < kMathEpsilon) continue;
      const Vector3f dv_w = {feats->feats.at(i).dv_x, feats->feats.at(i).dv_y,
                             feats->feats.at(i).dv_z};
      const Vector3f dv_c = world2cam(dv_w);
      if (dv_c.z() < kMathEpsilon) continue;
      const Vector2f& uv = cam2pixel(pt_c);
      if (!InFov(uv, half_patch_size_ * std::pow(2, optical_layer_ - 1))) continue;
      int32_t index = grid_index(uv);
      if (!feats->feats.at(i).is_line && pt_c.z() < grid_pdepth_.at(index)) {
        grid_pfeat_.at(index) = feats;
        pfeat_indice_.at(index) = i;
        grid_ptracked_.at(index) = true;
        grid_pscore_.at(index) = feats->feats.at(i).score;
        grid_pdepth_.at(index) = pt_c.z();
      } else if (feats->feats.at(i).is_line && pt_c.z() < grid_ldepth_.at(index)) {
        grid_lfeat_.at(index) = feats;
        lfeat_indice_.at(index) = i;
        grid_ltracked_.at(index) = true;
        grid_lscore_.at(index) = feats->feats.at(i).score;
        grid_ldepth_.at(index) = pt_c.z();
      }
    }
  }

  for (int32_t i = 0; i < grid_nums_; ++i) {
    if (grid_ptracked_.at(i) && grid_ltracked_.at(i)) {
      if (grid_pscore_.at(i) > grid_lscore_.at(i)) {
        grid_ltracked_.at(i) = false;
      } else {
        grid_ptracked_.at(i) = false;
      }
    }
  }

  double t3 = omp_get_wtime();

  t_vis_pre += (t2 - t1) * 1000.;
  t_vis_obs += (t3 - t2) * 1000.;
  LOG(INFO) << "[VisualPreprocess]: process image cost " << (t2 - t1) * 1000. << " ms, "
            << "Track visual feats from map cost " << (t3 - t2) * 1000. << " ms.";
}

bool CamHandler::Observation(const Ikfom::IkfomState& x, int32_t layer,
                             Eigen::MatrixXf* const h, Eigen::VectorXf* const r,
                             Eigen::VectorXf* const c) {
  if (!enable_) return false;
  assert(h != nullptr && r != nullptr && c != nullptr);
  assert(layer < optical_layer_);
  double t1 = omp_get_wtime();
  set_tf(x);

  // collect valid vis point observations
  std::vector<int32_t> pobs_index;
  pobs_index.reserve(grid_nums_);
  for (int32_t i = 0; i < grid_nums_; ++i) {
    if (grid_ptracked_.at(i)) {
      const VoxelVisualPtr& feats = grid_pfeat_.at(i);
      int32_t feat_index = pfeat_indice_.at(i);
      assert(!feats->feats.at(feat_index).is_line);
      const Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
      const Vector3f& pt_c = world2cam(pt_w);
      if (pt_c.z() < kMathEpsilon) {
        grid_ptracked_.at(i) = false;
        continue;
      }
      const Vector2f& uv = cam2pixel(pt_c);
      if (!InFov(uv, half_patch_size_ * std::pow(2, optical_layer_ - 1))) {
        grid_ptracked_.at(i) = false;
        continue;
      }
      pobs_index.emplace_back(i);
    }
  }
  // collect valid vis line observations
  std::vector<int32_t> lobs_index;
  lobs_index.reserve(grid_nums_);
  for (int32_t i = 0; i < grid_nums_; ++i) {
    if (grid_ltracked_.at(i)) {
      const VoxelVisualPtr& feats = grid_lfeat_.at(i);
      int32_t feat_index = lfeat_indice_.at(i);
      assert(feats->feats.at(feat_index).is_line);
      const Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
      const Vector3f& pt_c = world2cam(pt_w);
      if (pt_c.z() < kMathEpsilon) {
        grid_ltracked_.at(i) = false;
        continue;
      }
      const Vector2f& uv = cam2pixel(pt_c);
      if (!InFov(uv, half_patch_size_ * std::pow(2, optical_layer_ - 1))) {
        grid_ltracked_.at(i) = false;
        continue;
      }
      lobs_index.emplace_back(i);
    }
  }

  if (pobs_index.size() + lobs_index.size() == 0) {
    LOG(WARNING) << "[Visual Observation]: no effect vis feats";
    return false;
  }

  // PL-DVO: Point-Line Direct Visual Observation
  assert(half_patch_size_ % 2 == 0);
  int32_t ppatch_area = std::pow(2 * half_patch_size_ + 1, 2);
  int32_t lpatch_area = (half_patch_size_ + 1) * (4 * half_patch_size_ + 1);
  int32_t pobs_size = pobs_index.size() * ppatch_area;
  int32_t lobs_size = lobs_index.size() * lpatch_area;
  int32_t obs_size = pobs_size + lobs_size;
  float scale = std::pow(0.5, layer);
  *h = Eigen::MatrixXf::Zero(obs_size, 6);
  *r = Eigen::VectorXf::Zero(obs_size);
  *c = Eigen::VectorXf::Zero(obs_size);

  // compute residual and jacobian for point features
  double t2 = omp_get_wtime();
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(shared) shared(pobs_index, grid_pfeat_, pfeat_indice_, h, r, c)
#endif
  for (int32_t i = 0; i < pobs_index.size(); ++i) {
    const VoxelVisualPtr& feats = grid_pfeat_.at(pobs_index.at(i));
    int32_t feat_index = pfeat_indice_.at(pobs_index.at(i));
    assert(!feats->feats.at(feat_index).is_line);
    const Eigen::Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
    const Eigen::Vector3f& pt_c = world2cam(pt_w);
    const Eigen::Vector2f& uv = cam2pixel(pt_c);
    const std::vector<float>& ref_patch = feats->patches.at(feat_index).at(layer);
    assert(ref_patch.size() == ppatch_area);
    const Vector3f du_w = {feats->feats.at(feat_index).du_x,
                           feats->feats.at(feat_index).du_y,
                           feats->feats.at(feat_index).du_z};
    const Eigen::Vector3f& du_c = world2cam(du_w);
    const Eigen::Vector2f& du_uv = cam2pixel(du_c);
    const Eigen::Vector3f dv_w = {feats->feats.at(feat_index).dv_x,
                                  feats->feats.at(feat_index).dv_y,
                                  feats->feats.at(feat_index).dv_z};
    const Eigen::Vector3f& dv_c = world2cam(dv_w);
    const Eigen::Vector2f& dv_uv = cam2pixel(dv_c);
    Eigen::Matrix2f affine = Eigen::Matrix2f::Identity();
    affine.col(0) = du_uv - uv;
    affine.col(1) = dv_uv - uv;
    for (int32_t u = -half_patch_size_; u <= half_patch_size_; ++u) {
      for (int32_t v = -half_patch_size_; v <= half_patch_size_; ++v) {
        int32_t patch_index = (v + half_patch_size_) * (2 * half_patch_size_ + 1) +
                              (u + half_patch_size_);
        Eigen::Vector2f uv_patch = affine * Eigen::Vector2f(u, v);
        Eigen::Matrix<float, 1, 2> d_img_uv = Eigen::Matrix<float, 1, 2>::Zero();
        float l_ref = ref_patch.at(patch_index);
        float l_cur = InterpolateMat8u(img_pyr_.at(layer), uv * scale + uv_patch, &d_img_uv);
        int32_t obs_idx = i * ppatch_area + patch_index;
        assert(obs_idx < pobs_size);
        (*r)(obs_idx) = -(l_cur - l_ref);
        h->row(obs_idx) << scale * jacobian(d_img_uv, pt_c, x);
        (*c)(obs_idx) = meas_cov_;
      }
    }
  }
  // compute residual and jacobian for line features
  double t3 = omp_get_wtime();
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(shared) shared(lobs_index, grid_lfeat_, lfeat_indice_, h, r, c)
#endif
  for (int32_t i = 0; i < lobs_index.size(); ++i) {
    const VoxelVisualPtr& feats = grid_lfeat_.at(lobs_index.at(i));
    int32_t feat_index = lfeat_indice_.at(lobs_index.at(i));
    assert(feats->feats.at(feat_index).is_line);
    const Eigen::Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
    const Eigen::Vector3f& pt_c = world2cam(pt_w);
    const Eigen::Vector2f& uv = cam2pixel(pt_c);
    const std::vector<float>& ref_patch = feats->patches.at(feat_index).at(layer);
    assert(ref_patch.size() == lpatch_area);
    const Vector3f du_w = {feats->feats.at(feat_index).du_x,
                           feats->feats.at(feat_index).du_y,
                           feats->feats.at(feat_index).du_z};
    const Eigen::Vector3f& du_c = world2cam(du_w);
    const Eigen::Vector2f& du_uv = cam2pixel(du_c);
    const Eigen::Vector3f dv_w = {feats->feats.at(feat_index).dv_x,
                                  feats->feats.at(feat_index).dv_y,
                                  feats->feats.at(feat_index).dv_z};
    const Eigen::Vector3f& dv_c = world2cam(dv_w);
    const Eigen::Vector2f& dv_uv = cam2pixel(dv_c);
    Eigen::Matrix2f affine = Eigen::Matrix2f::Identity();
    affine.col(0) = du_uv - uv;
    affine.col(1) = dv_uv - uv;
    for (int32_t u = -half_patch_size_ * 2; u <= half_patch_size_ * 2; ++u) {
      for (int32_t v = -half_patch_size_ / 2; v <= half_patch_size_ / 2; ++v) {
        int32_t patch_index = (v + half_patch_size_ / 2) * (4 * half_patch_size_ + 1) +
                              (u + half_patch_size_ * 2);
        Eigen::Vector2f uv_patch = affine * Eigen::Vector2f(u, v);
        Eigen::Matrix<float, 1, 2> d_img_uv = Eigen::Matrix<float, 1, 2>::Zero();
        float l_ref = ref_patch.at(patch_index);
        float l_cur = InterpolateMat8u(img_pyr_.at(layer), uv * scale + uv_patch, &d_img_uv);
        int32_t obs_idx = pobs_size + i * lpatch_area + patch_index;
        assert(obs_idx < obs_size);
        (*r)(obs_idx) = -(l_cur - l_ref);
        h->row(obs_idx) << scale * jacobian(d_img_uv, pt_c, x);
        (*c)(obs_idx) = meas_cov_;
      }
    }
  }
  double t4 = omp_get_wtime();
  t_vis_obs += (t4 - t1) * 1000.;
  LOG(INFO) << "[Visual Observation]: fetch (" << pobs_index.size() << " points, "
            << lobs_index.size() << " lines) / " << grid_nums_ << " grids cost "
            << (t2 - t1) * 1000. << " ms, point observation cost " << (t3 - t2) * 1000.
            << " ms, line observation cost " << (t4 - t3) * 1000. << " ms, total "
            << (t4 - t1) * 1000. << " ms.";
  return true;
}

void CamHandler::RenderScan(const Ikfom::IkfomState& x,
                            const HybridCloud::Ptr& hybrid_scan) {
  double t1 = omp_get_wtime();
  scan_hybrid_ = hybrid_scan;
  img_depth_ = cv::Mat::zeros(img_rgb_.size(), CV_32FC1);
  scan_color_->clear();
  scan_color_->reserve(scan_hybrid_->size());
  for (auto& hybrid_point : hybrid_scan->points) {
    hybrid_point.colored = false;
    hybrid_point.score = 0.f;
    const Eigen::Vector3f& pt_c = world2cam(hybrid_point.getVector3fMap());
    if (pt_c.z() <= 0.1) continue;
    const Eigen::Vector2f& uv = cam2pixel(pt_c);
    if (!InFov(uv, 1)) continue;

    cv::Point pixel(uv.x(), uv.y());
    if (img_depth_.at<float>(pixel) <= 0. ||
        float(pt_c.z()) < img_depth_.at<float>(pixel)) {
      img_depth_.at<float>(pixel) = float(pt_c.z());
    }

    hybrid_point.colored = true;
    hybrid_point.score = shiTomasiScore(img_gray_, uv.cast<int32_t>());
    const Eigen::Vector3f& bgr = get_bgr(uv);
    hybrid_point.b = bgr(0);
    hybrid_point.g = bgr(1);
    hybrid_point.r = bgr(2);
    ColorPoint col_pt;
    col_pt.getVector3fMap() = hybrid_point.getVector3fMap();
    col_pt.getBGRVector3cMap() = hybrid_point.getBGRVector3cMap();
    scan_color_->emplace_back(col_pt);
  }
  double t2 = omp_get_wtime();
  LOG(INFO) << "[LiDAR MapIncremental]: Render Scan cost " << (t2 - t1) * 1000. << "ms";
}

VoxelVisualPtr CamHandler::MapIncremental(const Ikfom::IkfomState& x,
                                          const HybridCloud::Ptr& hybrid_scan) {
  double t1 = omp_get_wtime();
  vis_feats_->feats.clear();
  vis_feats_->patches.clear();
  vis_feats_->color_patches.clear();
  vis_feats_->feats.reserve(grid_nums_ * 2);
  vis_feats_->patches.reserve(grid_nums_ * 2);
  vis_feats_->color_patches.reserve(grid_nums_ * 2);

  // extract points and lines on image
  pix_points_.clear();
  cv::goodFeaturesToTrack(img_gray_, pix_points_, 200, 0.01, grid_size_ / 2);
  feat_points_.resize(pix_points_.size());
  int32_t num_extract_points = pix_points_.size();

  pix_lines_.clear();
  ed_->detectEdges(img_gray_);
  ed_->detectLines(pix_lines_);
  pix_lines_.resize(std::min(int32_t(pix_lines_.size()), 20));
  feat_lines_.resize(pix_lines_.size());
  int32_t num_extract_lines = pix_lines_.size();

  pix_samples_.resize(pix_lines_.size() * line_sample_num_);
  feat_samples_.resize(pix_lines_.size() * line_sample_num_);
  int32_t num_extract_samples = pix_samples_.size();

  // LM-VDE: LiDAR Map assisted Visual Depth Extraction
  double t2 = omp_get_wtime();
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(shared) shared(pix_points_, feat_points_, img_depth_)
#endif
  for (int32_t i = 0; i < pix_points_.size(); ++i) {
    feat_points_.at(i).is_valid = 0u;
    feat_points_.at(i).is_line = 0u;
    const cv::Point2f& uv_pixel = pix_points_.at(i);
    const Eigen::Vector2f uv(uv_pixel.x, uv_pixel.y);

    // Search nearest valid depth in 5x5 neighborhood
    double min_dist = 100.;
    float nearest_depth = -1.f;
    const int32_t half_neighbor = 5;
    for (int32_t u = -half_neighbor; u <= half_neighbor; ++u) {
      for (int32_t v = -half_neighbor; v <= half_neighbor; ++v) {
        cv::Point search_uv(uv_pixel.x + u, uv_pixel.y + v);
        if (search_uv.x < 0 || search_uv.x >= img_depth_.cols ||
            search_uv.y < 0 || search_uv.y >= img_depth_.rows) continue;
        float depth_val = img_depth_.at<float>(search_uv);
        if (depth_val <= kMathEpsilon) continue;
        double dist = std::hypot(u, v);
        if (dist < min_dist) {
          min_dist = dist;
          nearest_depth = depth_val;
        }
      }
    }
    if (nearest_depth <= kMathEpsilon) continue;

    Eigen::Vector3f query_c;
    query_c.x() = (uv.x() - cam_cx_) / cam_fx_ * nearest_depth;
    query_c.y() = (uv.y() - cam_cy_) / cam_fy_ * nearest_depth;
    query_c.z() = nearest_depth;
    Eigen::Vector3f query_w = cam2world(query_c);

    PointLMVDE(uv_pixel, query_w, feat_points_.at(i));
  }

  int32_t j = 0;
  for (int32_t i = 0; i < pix_points_.size(); ++i) {
    if (feat_points_.at(i).is_valid) {
      pix_points_.at(j) = pix_points_.at(i);
      feat_points_.at(j) = feat_points_.at(i);
      j++;
    }
  }
  pix_points_.resize(j);
  feat_points_.resize(j);
  int32_t num_lmvde_points = pix_points_.size();

  double t3 = omp_get_wtime();
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(shared) shared(pix_lines_, feat_lines_, img_depth_)
#endif
  for (int32_t i = 0; i < pix_lines_.size(); ++i) {
    feat_lines_.at(i).is_valid = 0u;
    feat_lines_.at(i).is_line = 1u;

    std::vector<Eigen::Vector3f> querys_w;
    std::vector<cv::Point2f> pix_samples;
    std::vector<VisualFeat> feat_samples;

    cv::Point2f pix_start(pix_lines_.at(i)[0], pix_lines_.at(i)[1]);
    cv::Point2f pix_end(pix_lines_.at(i)[2], pix_lines_.at(i)[3]);
    for (int32_t j = 0; j < 2 * line_sample_num_; ++j) {
      float ratio = float(j) / float(2 * line_sample_num_ - 1);
      cv::Point2f uv = pix_start * (1. - ratio) + pix_end * ratio;
      double min_dist = 100.;
      float nearest_depth = -1.f;
      const int32_t half_neighbor = 5;
      for (int32_t u = -half_neighbor; u <= half_neighbor; ++u) {
        for (int32_t v = -half_neighbor; v <= half_neighbor; ++v) {
          cv::Point search_uv(uv.x + u, uv.y + v);
          if (search_uv.x < 0 || search_uv.x >= img_depth_.cols ||
              search_uv.y < 0 || search_uv.y >= img_depth_.rows) continue;
          float depth_val = img_depth_.at<float>(search_uv);
          if (depth_val <= kMathEpsilon) continue;
          double dist = std::hypot(u, v);
          if (dist < min_dist) {
            min_dist = dist;
            nearest_depth = depth_val;
          }
        }
      }
      if (nearest_depth <= kMathEpsilon) continue;
      Eigen::Vector3f query_c;
      query_c.x() = (uv.x - cam_cx_) / cam_fx_ * nearest_depth;
      query_c.y() = (uv.y - cam_cy_) / cam_fy_ * nearest_depth;
      query_c.z() = nearest_depth;
      Eigen::Vector3f query_w = cam2world(query_c);

      pix_samples.emplace_back(uv);
      querys_w.emplace_back(query_w);
      feat_samples.emplace_back(VisualFeat());
    }

    bool status = LineLMVDE(pix_lines_.at(i), feat_lines_.at(i),
                            querys_w, pix_samples, feat_samples);
    if (status) {
      feat_lines_.at(i).is_valid = 1u;
    }

    for (int32_t j = 0, k = 0; j < line_sample_num_; ++j, ++k) {
      feat_samples_.at(i * line_sample_num_ + j).is_valid = 0u;
      if (k < feat_samples.size()) {
        pix_samples_.at(i * line_sample_num_ + j) = pix_samples.at(k);
        feat_samples_.at(i * line_sample_num_ + j) = feat_samples.at(k);
      }
    }
  }

  j = 0;
  for (int32_t i = 0; i < pix_lines_.size(); ++i) {
    if (feat_lines_.at(i).is_valid) {
      pix_lines_.at(j) = pix_lines_.at(i);
      feat_lines_.at(j) = feat_lines_.at(i);
      j++;
    }
  }
  pix_lines_.resize(j);
  feat_lines_.resize(j);
  int32_t num_lmvde_lines = pix_lines_.size();

  j = 0;
  for (int32_t i = 0; i < pix_samples_.size(); ++i) {
    if (feat_samples_.at(i).is_valid) {
      pix_samples_.at(j) = pix_samples_.at(i);
      feat_samples_.at(j) = feat_samples_.at(i);
      j++;
    }
  }
  pix_samples_.resize(j);
  feat_samples_.resize(j);
  int32_t num_lmvde_samples = pix_samples_.size();

  double t4 = omp_get_wtime();

  std::vector<int32_t> points_indice(grid_nums_, -1);
  for (int32_t i = 0; i < pix_points_.size(); ++i) {
    const cv::Point2f& corner = pix_points_.at(i);
    const Eigen::Vector2f uv(corner.x, corner.y);
    const float score = shiTomasiScore(img_gray_, uv.cast<int32_t>());
    feat_points_.at(i).score = score;
    int32_t index = grid_index(uv);
    if (!grid_ptracked_.at(index) && score > grid_pscore_.at(index)) {
      grid_pscore_.at(index) = score;
      grid_pupdate_.at(index) = true;
      points_indice.at(index) = i;
    }
  }

  std::vector<int32_t> samples_indice(grid_nums_, -1);
  for (int32_t i = 0; i < pix_samples_.size(); ++i) {
    const cv::Point2f& sample = pix_samples_.at(i);
    const Eigen::Vector2f uv(sample.x, sample.y);
    const float score = shiTomasiScore(img_gray_, uv.cast<int32_t>());
    feat_samples_.at(i).score = score;
    int32_t index = grid_index(uv);
    if (!grid_ltracked_.at(index) && score > grid_lscore_.at(index)) {
      grid_lscore_.at(index) = score;
      grid_lupdate_.at(index) = true;
      samples_indice.at(index) = i;
    }
  }

  double t5 = omp_get_wtime();

#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int32_t i = 0; i < grid_nums_; ++i) {
    if (grid_ptracked_.at(i)) {
      const VoxelVisualPtr& feats = grid_pfeat_.at(i);
      int32_t feat_index = pfeat_indice_.at(i);
      const Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
      const Vector3f& pt_c = world2cam(pt_w);
      const Vector2f& uv = cam2pixel(pt_c);
      float score = shiTomasiScore(img_gray_, uv.cast<int32_t>());
      if (score > feats->feats.at(feat_index).score) {
        bool status = PointLMVDE(cv::Point2f(uv.x(), uv.y()), pt_w,
                                 feats->feats.at(feat_index));
        if (status) {
          feats->feats.at(feat_index).score = score;
          get_pfeatpatch(uv, feats->patches.at(feat_index));
          get_pcolorpatch(uv, feats->color_patches.at(feat_index));
        }
      }
    }
    if (grid_ltracked_.at(i)) {
      // const VoxelVisualPtr& feats = grid_lfeat_.at(i);
      // int32_t feat_index = lfeat_indice_.at(i);
      // const Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
      // const Vector3f& pt_c = world2cam(pt_w);
      // const Vector2f& uv = cam2pixel(pt_c);
      // float score = shiTomasiScore(img_gray_, uv.cast<int32_t>());
      // if (score > feats->feats.at(feat_index).score) {
      //   feats->feats.at(feat_index).score = score;
      //   const Vector3f du_w = {feats->feats.at(feat_index).du_x,
      //                          feats->feats.at(feat_index).du_y,
      //                          feats->feats.at(feat_index).du_z};
      //   const Vector3f& du_c = world2cam(du_w);
      //   const Vector2f& du_uv = cam2pixel(du_c);
      //   const Vector3f dv_w = {feats->feats.at(feat_index).dv_x,
      //                          feats->feats.at(feat_index).dv_y,
      //                          feats->feats.at(feat_index).dv_z};
      //   const Vector3f& dv_c = world2cam(dv_w);
      //   const Vector2f& dv_uv = cam2pixel(dv_c);
      //   Eigen::Matrix2f affine = Eigen::Matrix2f::Identity();
      //   affine.col(0) = du_uv - uv;
      //   affine.col(1) = dv_uv - uv;
      //   get_pfeatpatch(uv, feats->patches.at(feat_index), affine);         // todo
      //   get_pcolorpatch(uv, feats->color_patches.at(feat_index), affine);  // todo
      // }
    }
  }

  for (int32_t i = 0; i < grid_nums_; ++i) {
    if (grid_pupdate_.at(i)) {
      int32_t feat_index = points_indice.at(i);
      const VisualFeat& point = feat_points_.at(feat_index);
      const Eigen::Vector2f uv = {pix_points_.at(feat_index).x,
                                  pix_points_.at(feat_index).y};
      FeatPatch patch;
      get_pfeatpatch(uv, patch);
      ColorPatch color_patch;
      get_pcolorpatch(uv, color_patch);
      vis_feats_->feats.emplace_back(point);
      vis_feats_->patches.emplace_back(patch);
      vis_feats_->color_patches.emplace_back(color_patch);
    }
    if (grid_lupdate_.at(i)) {
      int32_t feat_index = samples_indice.at(i);
      const VisualFeat& sample = feat_samples_.at(feat_index);
      const Eigen::Vector2f uv = {pix_samples_.at(feat_index).x,
                                  pix_samples_.at(feat_index).y};
      const Eigen::Vector3f du_w = {sample.du_x, sample.du_y, sample.du_z};
      const Eigen::Vector3f du_c = world2cam(du_w);
      const Eigen::Vector2f du_uv = cam2pixel(du_c);
      const Eigen::Vector3f dv_w = {sample.dv_x, sample.dv_y, sample.dv_z};
      const Eigen::Vector3f dv_c = world2cam(dv_w);
      const Eigen::Vector2f dv_uv = cam2pixel(dv_c);
      Eigen::Matrix2f affine = Eigen::Matrix2f::Identity();
      affine.col(0) = du_uv - uv;
      affine.col(1) = dv_uv - uv;
      FeatPatch patch;
      get_lfeatpatch(uv, patch, affine);
      ColorPatch color_patch;
      get_lcolorpatch(uv, color_patch, affine);
      vis_feats_->feats.emplace_back(sample);
      vis_feats_->patches.emplace_back(patch);
      vis_feats_->color_patches.emplace_back(color_patch);
    }
  }
  double t6 = omp_get_wtime();

  LOG(INFO) << "[Visual MapIncremental] cost " << (t6 - t1) * 1000. << "ms, Extract "
            << num_extract_points << " / " << num_extract_lines << "  PL feats cost "
            << (t2 - t1) * 1000. << "ms, \nLM-VDE: " << num_lmvde_points
            << " Points cost " << (t3 - t2) * 1000. << "ms, " << num_lmvde_lines
            << " Lines cost " << (t4 - t3) * 1000. << "ms, \n"
            << "assign to grids cost " << (t5 - t4) * 1000. << "ms, "
            << "update grid cost " << (t6 - t5) * 1000. << "ms.";

  if (pcd_save_en_) *scan_save_ += *scan_color_;
  return vis_feats_;
}

bool CamHandler::PointLMVDE(const cv::Point2f& pix_point,
                            const Eigen::Vector3f& query_w,
                            VisualFeat& feat_point) const {
  feat_point.is_valid = 0u;

  LIVOPoint query_pt;
  query_pt.x = query_w.x();
  query_pt.y = query_w.y();
  query_pt.z = query_w.z();
  VoxelSurfelConstPtr near_surf = local_map_->GetNearestSurfel(query_pt);
  if (near_surf == nullptr || near_surf->sigma[0] > 0.05) return false;

  // Ray-surfel intersection for refined depth
  Eigen::Vector3f surf_center_c = world2cam(near_surf->center);
  Eigen::Vector3f surf_normal_c = tf_wc_.rot().conjugate() * near_surf->normal;
  Eigen::Vector3f ray_dir = {(pix_point.x - cam_cx_) / cam_fx_,
                             (pix_point.y - cam_cy_) / cam_fy_, 1.f};
  Eigen::Vector3f ray_du_dir{(pix_point.x + 1 - cam_cx_) / cam_fx_,
                             (pix_point.y - cam_cy_) / cam_fy_, 1.f};
  Eigen::Vector3f ray_dv_dir{(pix_point.x - cam_cx_) / cam_fx_,
                             (pix_point.y + 1 - cam_cy_) / cam_fy_, 1.f};
  ray_dir.normalize();
  ray_du_dir.normalize();
  ray_dv_dir.normalize();

  // Ray: P = t * ray_dir, Plane: (P - surf_center_c) · surf_normal_c = 0
  float denom = ray_dir.dot(surf_normal_c);
  if (std::abs(denom) < kMathEpsilon) return false;
  float t = surf_center_c.dot(surf_normal_c) / denom;
  if (t <= kMathEpsilon) return false;

  float du_denom = ray_du_dir.dot(surf_normal_c);
  if (std::abs(du_denom) < kMathEpsilon) return false;
  float du_t = surf_center_c.dot(surf_normal_c) / du_denom;
  if (du_t <= kMathEpsilon) return false;

  float dv_denom = ray_dv_dir.dot(surf_normal_c);
  if (std::abs(dv_denom) < kMathEpsilon) return false;
  float dv_t = surf_center_c.dot(surf_normal_c) / dv_denom;
  if (dv_t <= kMathEpsilon) return false;

  Eigen::Vector3f intersect_c = t * ray_dir;
  Eigen::Vector3f intersect_w = cam2world(intersect_c);
  Eigen::Vector3f intersect_du_c = du_t * ray_du_dir;
  Eigen::Vector3f intersect_du_w = cam2world(intersect_du_c);
  Eigen::Vector3f intersect_dv_c = dv_t * ray_dv_dir;
  Eigen::Vector3f intersect_dv_w = cam2world(intersect_dv_c);

  float scale = half_patch_size_ * std::pow(2.f, optical_layer_ - 1);

  if ((intersect_w - near_surf->center).norm() > 3 * near_surf->sigma[1] ||
      (intersect_du_w - intersect_w).norm() * scale > 3 * near_surf->sigma[1] ||
      (intersect_dv_w - intersect_w).norm() * scale > 3 * near_surf->sigma[1])
    return false;

  feat_point.x = intersect_w.x();
  feat_point.y = intersect_w.y();
  feat_point.z = intersect_w.z();
  feat_point.getNormalVector3fMap() = ray_dir;

  feat_point.du_x = intersect_du_w.x();
  feat_point.du_y = intersect_du_w.y();
  feat_point.du_z = intersect_du_w.z();

  feat_point.dv_x = intersect_dv_w.x();
  feat_point.dv_y = intersect_dv_w.y();
  feat_point.dv_z = intersect_dv_w.z();

  Eigen::Vector3f bgr = get_bgr(Eigen::Vector2f(pix_point.x, pix_point.y));
  feat_point.b = uint8_t(bgr(0));
  feat_point.g = uint8_t(bgr(1));
  feat_point.r = uint8_t(bgr(2));

  feat_point.is_valid = 1u;

  return true;
}

bool CamHandler::LineLMVDE(const cv::Vec4f& pix_line, VisualFeat& feat_line,
                           const std::vector<Eigen::Vector3f>& querys_w,
                           std::vector<cv::Point2f>& pix_samples,
                           std::vector<VisualFeat>& feat_samples) const {
  assert(pix_samples.size() == querys_w.size());
  assert(feat_samples.size() == querys_w.size());

  feat_line.is_valid = 0u;

  if (querys_w.size() < line_sample_num_) {
    pix_samples.clear();
    feat_samples.clear();
    return false;
  }

  Eigen::Vector2f pix_du = {pix_line[2] - pix_line[0],
                            pix_line[3] - pix_line[1]};
  Eigen::Matrix2f rot;
  rot << 0, -1, 1, 0;
  Eigen::Vector2f pix_dv = rot * pix_du;
  pix_du.normalize();
  pix_dv.normalize();

  for (int32_t i = 0; i < querys_w.size(); ++i) {
    feat_samples.at(i).is_valid = 0u;
    feat_samples.at(i).is_line = 1u;

    LIVOPoint query_pt;
    query_pt.x = querys_w.at(i).x();
    query_pt.y = querys_w.at(i).y();
    query_pt.z = querys_w.at(i).z();
    VoxelSurfelConstPtr near_surf = local_map_->GetNearestSurfel(query_pt);
    if (near_surf == nullptr || near_surf->sigma[0] > 0.01) continue;

    Eigen::Vector2f uv = {pix_samples.at(i).x, pix_samples.at(i).y};
    Eigen::Vector3f surf_center_c = world2cam(near_surf->center);
    Eigen::Vector3f surf_normal_c = tf_wc_.rot().conjugate() * near_surf->normal;
    Eigen::Vector3f ray_dir = {(uv.x() - cam_cx_) / cam_fx_,
                               (uv.y() - cam_cy_) / cam_fy_, 1.f};
    Eigen::Vector3f ray_du_dir{(uv.x() + pix_du.x() - cam_cx_) / cam_fx_,
                               (uv.y() + pix_du.y() - cam_cy_) / cam_fy_, 1.f};
    Eigen::Vector3f ray_dv_dir{(uv.x() + pix_dv.x() - cam_cx_) / cam_fx_,
                               (uv.y() + pix_dv.y() - cam_cy_) / cam_fy_, 1.f};
    ray_dir.normalize();
    ray_du_dir.normalize();
    ray_dv_dir.normalize();

    // Ray: P = t * ray_dir, Plane: (P - surf_center_c) · surf_normal_c = 0
    float denom = ray_dir.dot(surf_normal_c);
    if (std::abs(denom) < kMathEpsilon) continue;
    float t = surf_center_c.dot(surf_normal_c) / denom;
    if (t <= kMathEpsilon) continue;

    float du_denom = ray_du_dir.dot(surf_normal_c);
    if (std::abs(du_denom) < kMathEpsilon) continue;
    float du_t = surf_center_c.dot(surf_normal_c) / du_denom;
    if (du_t <= kMathEpsilon) continue;

    float dv_denom = ray_dv_dir.dot(surf_normal_c);
    if (std::abs(dv_denom) < kMathEpsilon) continue;
    float dv_t = surf_center_c.dot(surf_normal_c) / dv_denom;
    if (dv_t <= kMathEpsilon) continue;

    Eigen::Vector3f intersect_c = t * ray_dir;
    Eigen::Vector3f intersect_w = cam2world(intersect_c);
    Eigen::Vector3f intersect_du_c = du_t * ray_du_dir;
    Eigen::Vector3f intersect_du_w = cam2world(intersect_du_c);
    Eigen::Vector3f intersect_dv_c = dv_t * ray_dv_dir;
    Eigen::Vector3f intersect_dv_w = cam2world(intersect_dv_c);

    float scale_du = half_patch_size_ * 2 * std::pow(2.f, optical_layer_ - 1);
    float scale_dv = half_patch_size_ / 2 * std::pow(2.f, optical_layer_ - 1);

    if ((intersect_w - near_surf->center).norm() > 3 * near_surf->sigma[1] ||
        (intersect_du_w - intersect_w).norm() * scale_du > 3 * near_surf->sigma[1] ||
        (intersect_dv_w - intersect_w).norm() * scale_dv > 3 * near_surf->sigma[1])
      continue;

    feat_samples.at(i).x = intersect_w.x();
    feat_samples.at(i).y = intersect_w.y();
    feat_samples.at(i).z = intersect_w.z();
    feat_samples.at(i).getNormalVector3fMap() = ray_dir;

    feat_samples.at(i).du_x = intersect_du_w.x();
    feat_samples.at(i).du_y = intersect_du_w.y();
    feat_samples.at(i).du_z = intersect_du_w.z();

    feat_samples.at(i).dv_x = intersect_dv_w.x();
    feat_samples.at(i).dv_y = intersect_dv_w.y();
    feat_samples.at(i).dv_z = intersect_dv_w.z();

    Eigen::Vector3f bgr = get_bgr(Eigen::Vector2f(uv.x(), uv.y()));
    feat_samples.at(i).b = uint8_t(bgr(0));
    feat_samples.at(i).g = uint8_t(bgr(1));
    feat_samples.at(i).r = uint8_t(bgr(2));

    feat_samples.at(i).is_valid = 1u;
  }

  int32_t j = 0;
  for (int32_t i = 0; i < feat_samples.size(); ++i) {
    if (feat_samples.at(i).is_valid) {
      pix_samples.at(j) = pix_samples.at(i);
      feat_samples.at(j) = feat_samples.at(i);
      j++;
    }
  }
  pix_samples.resize(j);
  feat_samples.resize(j);
  if (pix_samples.size() < line_sample_num_) {
    return false;
  }

  feat_line.is_valid = 1u;
  feat_line.x = feat_samples.front().x;
  feat_line.y = feat_samples.front().y;
  feat_line.z = feat_samples.front().z;
  feat_line.normal_x = feat_samples.back().x;
  feat_line.normal_y = feat_samples.back().y;
  feat_line.normal_z = feat_samples.back().z;

  return true;
}

Vector3f CamHandler::world2cam(const Vector3f& pt_w) const {
  return tf_wc_.inverse() * pt_w;
}

Vector3f CamHandler::cam2world(const Vector3f& pt_c) const {
  return tf_wc_ * pt_c;
}

Vector2f CamHandler::cam2pixel(const Vector3f& pt_c) const {
  assert(pt_c.z() > kMathEpsilon);
  const double x = pt_c.x() / pt_c.z();
  const double y = pt_c.y() / pt_c.z();
  return Vector2f(x * cam_fx_ + cam_cx_, y * cam_fy_ + cam_cy_);
}

int32_t CamHandler::grid_index(const Vector2f& uv) const {
  return int32_t(uv.x() / grid_size_) +
         int32_t(uv.y() / grid_size_) * (width_ / grid_size_ + 1);
}

bool CamHandler::InFov(const Vector2f& uv, int32_t boundry) const {
  const Vector2i& obs = uv.cast<int32_t>();
  if (obs.x() >= boundry && obs.x() < width_ - boundry &&
      obs.y() >= boundry && obs.y() < height_ - boundry) {
    return true;
  }
  return false;
}

void CamHandler::set_tf(const Ikfom::IkfomState& state) {
  tf_wc_ = se3::SE3(state.rot.matrix() * ric_,
                    state.rot.rot() * tic_ + state.pos);
}

se3::SE3 CamHandler::pose(const Ikfom::IkfomState& x) {
  set_tf(x);
  return tf_wc_;
}

void CamHandler::update_pose(Ikfom::IkfomState& x, const se3::SE3& pose) {
  Eigen::Matrix3f rwc = pose.rotation_matrix();
  Eigen::Vector3f twc = pose.translation();

  Eigen::Matrix3f rwi = rwc * ric_.transpose();
  Eigen::Vector3f twi = twc - rwi * tic_;

  x.rot = mtk::SO3(rwi);
  x.pos = twi;
}

Eigen::Vector3f CamHandler::get_bgr(const Vector2f& uv) const {
  const float u_ref = uv[0];
  const float v_ref = uv[1];
  const int u_ref_i = floorf(uv[0]);
  const int v_ref_i = floorf(uv[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t* img_ptr = (uint8_t*)img_rgb_.data + ((v_ref_i)*width_ + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] +
            w_ref_tr * img_ptr[0 + 3] +
            w_ref_bl * img_ptr[width_ * 3] +
            w_ref_br * img_ptr[width_ * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] +
            w_ref_tr * img_ptr[1 + 3] +
            w_ref_bl * img_ptr[1 + width_ * 3] +
            w_ref_br * img_ptr[width_ * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] +
            w_ref_tr * img_ptr[2 + 3] +
            w_ref_bl * img_ptr[2 + width_ * 3] +
            w_ref_br * img_ptr[width_ * 3 + 2 + 3];
  Eigen::Vector3f pixel(B, G, R);
  return pixel;
}

Eigen::Vector3f CamHandler::get_rgb(Eigen::Vector3f point) const {
  const Eigen::Vector3f& pt_c = world2cam(point);
  if (pt_c.z() <= 0.1) return -Eigen::Vector3f::Ones();
  const Eigen::Vector2f& uv = cam2pixel(pt_c);
  if (!InFov(uv, 1)) return -Eigen::Vector3f::Ones();
  const Eigen::Vector3f& bgr = get_bgr(uv);
  return {bgr.z(), bgr.y(), bgr.x()};
}

float CamHandler::InterpolateMat8u(const cv::Mat& img, const Vector2f& uv,
                                   Matrix<float, 1, 2>* const jacobian) const {
  assert(img.type() == CV_8U);

  const int32_t u_floor = floorf(uv.x());
  const int32_t v_floor = floorf(uv.y());

  if (u_floor < 0 || u_floor + 1 >= img.cols ||
      v_floor < 0 || v_floor + 1 >= img.rows) {
    if (jacobian != nullptr) (*jacobian).setZero();
    return 0.f;
  }

  const double subpix_x = uv.x() - u_floor;
  const double subpix_y = uv.y() - v_floor;
  const double w00 = (1.0 - subpix_x) * (1.0 - subpix_y);
  const double w01 = subpix_x * (1.0 - subpix_y);
  const double w10 = (1.0 - subpix_x) * subpix_y;
  const double w11 = subpix_x * subpix_y;
  const int stride = img.cols;
  const uint8_t* img_ptr = img.ptr<uint8_t>() + v_floor * stride + u_floor;
  if (jacobian != nullptr) {
    const double luminosity_right = w00 * img_ptr[1] +
                                    w01 * img_ptr[2] +
                                    w10 * img_ptr[stride + 1] +
                                    w11 * img_ptr[stride + 2 * 1];
    const double luminosity_left = w00 * img_ptr[-1] +
                                   w01 * img_ptr[0] +
                                   w10 * img_ptr[stride - 1] +
                                   w11 * img_ptr[stride];
    const double du = 0.5 * (luminosity_right - luminosity_left);
    const double luminosity_bottom = w00 * img_ptr[stride] +
                                     w01 * img_ptr[stride + 1] +
                                     w10 * img_ptr[stride * 2] +
                                     w11 * img_ptr[stride * 2 + 1];
    const double luminosity_top = w00 * img_ptr[-stride] +
                                  w01 * img_ptr[-stride + 1] +
                                  w10 * img_ptr[0] +
                                  w11 * img_ptr[1];
    const double dv = 0.5 * (luminosity_bottom - luminosity_top);
    (*jacobian) << du, dv;
  }
  return w00 * img_ptr[0] +
         w01 * img_ptr[1] +
         w10 * img_ptr[stride] +
         w11 * img_ptr[stride + 1];
}

Eigen::Matrix<float, 1, 6> CamHandler::jacobian(const Matrix<float, 1, 2>& d_img_uv,
                                                const Vector3f& pt_c,
                                                const Ikfom::IkfomState& x) const {
  assert(pt_c.z() > kMathEpsilon);
  Eigen::Matrix<float, 1, 6> jaco = Eigen::Matrix<float, 1, 6>::Zero();
  Eigen::Matrix<float, 2, 3> d_uv_ptc = Eigen::Matrix<float, 2, 3>::Zero();
  const double z_inv = 1. / pt_c.z();
  const double z_inv_square = z_inv * z_inv;
  d_uv_ptc(0, 0) = cam_fx_ * z_inv;
  d_uv_ptc(0, 2) = -cam_fx_ * pt_c.x() * z_inv_square;
  d_uv_ptc(1, 1) = cam_fy_ * z_inv;
  d_uv_ptc(1, 2) = -cam_fy_ * pt_c.y() * z_inv_square;
  const Eigen::Matrix3f& rci = ric_.transpose();
  const Eigen::Matrix3f& d_ptc_rot = mtk::SO3::hat(pt_c) * rci + rci * mtk::SO3::hat(tic_);
  const Eigen::Matrix3f& d_ptc_pos = -rci * x.rot.inverse().matrix();
  jaco.block<1, 3>(0, 0) = d_img_uv * d_uv_ptc * d_ptc_rot;
  jaco.block<1, 3>(0, 3) = d_img_uv * d_uv_ptc * d_ptc_pos;
  return jaco;
}

void CamHandler::get_pfeatpatch(const Eigen::Vector2f& uv, FeatPatch& patch,
                                const Eigen::Matrix2f& affine) const {
  patch.resize(optical_layer_);
  for (int32_t l = 0; l < optical_layer_; ++l) {
    float scale = std::pow(0.5, l);
    int32_t ppatch_area = std::pow(2 * half_patch_size_ + 1, 2);
    patch.at(l).resize(ppatch_area);
    for (int32_t u = -half_patch_size_; u <= half_patch_size_; ++u) {
      for (int32_t v = -half_patch_size_; v <= half_patch_size_; ++v) {
        int32_t patch_index = (v + half_patch_size_) * (2 * half_patch_size_ + 1) +
                              (u + half_patch_size_);
        Vector2f uv_patch{u, v};
        uv_patch = affine * uv_patch;
        float l_cur = InterpolateMat8u(img_pyr_.at(l), uv * scale + uv_patch);
        patch.at(l).at(patch_index) = l_cur;
      }
    }
  }
}

void CamHandler::get_lfeatpatch(const Eigen::Vector2f& uv, FeatPatch& patch,
                                const Eigen::Matrix2f& affine) const {
  patch.resize(optical_layer_);
  for (int32_t l = 0; l < optical_layer_; ++l) {
    float scale = std::pow(0.5, l);
    int32_t lpatch_area = (half_patch_size_ + 1) * (4 * half_patch_size_ + 1);
    patch.at(l).resize(lpatch_area);
    for (int32_t u = -half_patch_size_ * 2; u <= half_patch_size_ * 2; ++u) {
      for (int32_t v = -half_patch_size_ / 2; v <= half_patch_size_ / 2; ++v) {
        int32_t patch_index = (v + half_patch_size_ / 2) * (4 * half_patch_size_ + 1) +
                              (u + half_patch_size_ * 2);
        Vector2f uv_patch{u, v};
        uv_patch = affine * uv_patch;
        float l_cur = InterpolateMat8u(img_pyr_.at(l), uv * scale + uv_patch);
        patch.at(l).at(patch_index) = l_cur;
      }
    }
  }
}

void CamHandler::get_pcolorpatch(const Eigen::Vector2f& uv, ColorPatch& patch,
                                 const Eigen::Matrix2f& affine) const {
  patch.resize(3);
  std::vector<cv::Mat> channels;
  cv::split(img_rgb_, channels);
  float scale = std::pow(2, optical_layer_);
  int32_t shalf_patch_size = half_patch_size_ * scale;
  for (int32_t c = 0; c < 3; ++c) {
    int32_t ppatch_area = std::pow(2 * shalf_patch_size + 1, 2);
    patch.at(c).resize(ppatch_area);
    for (int32_t u = -shalf_patch_size; u <= shalf_patch_size; ++u) {
      for (int32_t v = -shalf_patch_size; v <= shalf_patch_size; ++v) {
        int32_t patch_index = (v + shalf_patch_size) * (2 * shalf_patch_size + 1) +
                              (u + shalf_patch_size);
        Vector2f uv_patch{u, v};
        uv_patch = affine * uv_patch;
        float l_cur = InterpolateMat8u(channels.at(c), uv + uv_patch);
        patch.at(c).at(patch_index) = l_cur;
      }
    }
  }
}

void CamHandler::get_lcolorpatch(const Eigen::Vector2f& uv, ColorPatch& patch,
                                 const Eigen::Matrix2f& affine) const {
  patch.resize(3);
  std::vector<cv::Mat> channels;
  cv::split(img_rgb_, channels);
  float scale = std::pow(2, optical_layer_);
  int32_t shalf_patch_size = half_patch_size_ * scale;
  for (int32_t c = 0; c < 3; ++c) {
    int32_t lpatch_area = (shalf_patch_size + 1) * (4 * shalf_patch_size + 1);
    patch.at(c).resize(lpatch_area);
    for (int32_t u = -shalf_patch_size * 2; u <= shalf_patch_size * 2; ++u) {
      for (int32_t v = -shalf_patch_size / 2; v <= shalf_patch_size / 2; ++v) {
        int32_t patch_index = (v + shalf_patch_size / 2) * (4 * shalf_patch_size + 1) +
                              (u + shalf_patch_size * 2);
        Vector2f uv_patch{u, v};
        uv_patch = affine * uv_patch;
        float l_cur = InterpolateMat8u(channels.at(c), uv + uv_patch);
        patch.at(c).at(patch_index) = l_cur;
      }
    }
  }
}

float CamHandler::shiTomasiScore(const cv::Mat& img, const Eigen::Vector2i& uv) {
  float dXX = 0.f;
  float dYY = 0.f;
  float dXY = 0.f;
  const int32_t halfbox_size = 4;
  const int32_t box_size = 2 * halfbox_size;
  const int32_t box_area = box_size * box_size;
  const int32_t x_min = uv.x() - halfbox_size;
  const int32_t x_max = uv.x() + halfbox_size;
  const int32_t y_min = uv.y() - halfbox_size;
  const int32_t y_max = uv.y() + halfbox_size;
  if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1) {
    return 0.f;  // patch is too close to the boundary
  }
  const int32_t stride = img.cols;
  for (int32_t y = y_min; y < y_max; ++y) {
    const uint8_t* ptr_left = img.ptr<uint8_t>() + stride * y + x_min - 1;
    const uint8_t* ptr_right = img.ptr<uint8_t>() + stride * y + x_min + 1;
    const uint8_t* ptr_top = img.ptr<uint8_t>() + stride * (y - 1) + x_min;
    const uint8_t* ptr_bottom = img.ptr<uint8_t>() + stride * (y + 1) + x_min;
    for (int32_t x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
      float dx = *ptr_right - *ptr_left;
      float dy = *ptr_bottom - *ptr_top;
      dXX += dx * dx;
      dYY += dy * dy;
      dXY += dx * dy;
    }
  }
  // Find and return smaller eigenvalue:
  float box_area_inv = 1.f / box_area;
  dXX = dXX * 0.5f * box_area_inv;
  dYY = dYY * 0.5f * box_area_inv;
  dXY = dXY * 0.5f * box_area_inv;
  return 0.5f * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}

void CamHandler::Publish() {
  const auto& gray_type = sensor_msgs::image_encodings::MONO8;
  const auto& rgb_type = sensor_msgs::image_encodings::BGR8;
  if (pub_feat_.getNumSubscribers() > 0) {
    cv::Mat feat_img = img_rgb_.clone();
    for (int32_t col = 0; col < width_; col += grid_size_)
      cv::line(feat_img, cv::Point(col, 0), cv::Point(col, height_ - 1), cv::Scalar(255, 255, 255), 1);
    for (int32_t row = 0; row < width_; row += grid_size_)
      cv::line(feat_img, cv::Point(0, row), cv::Point(width_ - 1, row), cv::Scalar(255, 255, 255), 1);
    for (int32_t i = 0; i < grid_nums_; ++i) {
      if (grid_ptracked_.at(i)) {
        const VoxelVisualPtr& feats = grid_pfeat_.at(i);
        int32_t feat_index = pfeat_indice_.at(i);
        const Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
        const Vector3f& pt_c = world2cam(pt_w);
        const Vector2f& uv = cam2pixel(pt_c);
        cv::circle(feat_img, cv::Point2f(uv.x(), uv.y()), 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
      }
    }
    for (const auto& point : vis_feats_->feats) {
      const Eigen::Vector3f& pt_c = world2cam(point.getVector3fMap());
      if (pt_c.z() <= kMathEpsilon) continue;
      const Eigen::Vector2f& uv = cam2pixel(pt_c);
      cv::circle(feat_img, cv::Point2f(uv.x(), uv.y()), 2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(cur_cam_header_, rgb_type, feat_img).toImageMsg();
    pub_feat_.publish(msg);
  }

  if (pub_patch_.getNumSubscribers() > 0) {
    cv::Mat patch_img = cv::Mat::zeros(img_rgb_.size(), CV_8UC3);
    for (int32_t i = 0; i < grid_nums_; ++i) {
      if (grid_ptracked_.at(i)) {
        const VoxelVisualPtr& feats = grid_pfeat_.at(i);
        int32_t feat_index = pfeat_indice_.at(i);
        const Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
        const Vector3f& pt_c = world2cam(pt_w);
        const Vector2f& uv = cam2pixel(pt_c);
        if (!InFov(uv, half_patch_size_)) continue;
        const Vector3f du_w = {feats->feats.at(feat_index).du_x,
                               feats->feats.at(feat_index).du_y,
                               feats->feats.at(feat_index).du_z};
        const Vector3f& du_c = world2cam(du_w);
        const Vector2f& du_uv = cam2pixel(du_c);
        const Vector3f dv_w = {feats->feats.at(feat_index).dv_x,
                               feats->feats.at(feat_index).dv_y,
                               feats->feats.at(feat_index).dv_z};
        const Vector3f& dv_c = world2cam(dv_w);
        const Vector2f& dv_uv = cam2pixel(dv_c);
        Eigen::Matrix2f affine = Eigen::Matrix2f::Identity();
        affine.col(0) = du_uv - uv;
        affine.col(1) = dv_uv - uv;
        float scale = std::pow(2, optical_layer_);
        int32_t shalf_patch_size = half_patch_size_ * scale;
        const ColorPatch& color_patch = feats->color_patches.at(feat_index);
        for (int32_t u = -shalf_patch_size; u <= shalf_patch_size; ++u) {
          for (int32_t v = -shalf_patch_size; v <= shalf_patch_size; ++v) {
            int32_t patch_index = (v + shalf_patch_size) * (2 * shalf_patch_size + 1) +
                                  (u + shalf_patch_size);
            Vector2f uv_patch{u, v};
            uv_patch = affine * uv_patch;
            Vector2f pixel_pos = uv + uv_patch;
            int32_t x = int32_t(pixel_pos.x() + 0.5f);
            int32_t y = int32_t(pixel_pos.y() + 0.5f);
            if (x >= 0 && x < patch_img.cols && y >= 0 && y < patch_img.rows) {
              uint8_t b_val = uint8_t(color_patch.at(0).at(patch_index));
              uint8_t g_val = uint8_t(color_patch.at(1).at(patch_index));
              uint8_t r_val = uint8_t(color_patch.at(2).at(patch_index));
              cv::Vec3b& pixel = patch_img.at<cv::Vec3b>(y, x);
              pixel[0] = b_val;
              pixel[1] = g_val;
              pixel[2] = r_val;
            }
          }
        }
        cv::circle(patch_img, cv::Point2f(uv.x(), uv.y()), 2, cv::Scalar(255, 0, 0), -1);
      }
      if (grid_ltracked_.at(i)) {
        const VoxelVisualPtr& feats = grid_lfeat_.at(i);
        int32_t feat_index = lfeat_indice_.at(i);
        const Vector3f& pt_w = feats->feats.at(feat_index).getVector3fMap();
        const Vector3f& pt_c = world2cam(pt_w);
        const Vector2f& uv = cam2pixel(pt_c);
        if (!InFov(uv, half_patch_size_)) continue;
        const Vector3f du_w = {feats->feats.at(feat_index).du_x,
                               feats->feats.at(feat_index).du_y,
                               feats->feats.at(feat_index).du_z};
        const Vector3f& du_c = world2cam(du_w);
        const Vector2f& du_uv = cam2pixel(du_c);
        const Vector3f dv_w = {feats->feats.at(feat_index).dv_x,
                               feats->feats.at(feat_index).dv_y,
                               feats->feats.at(feat_index).dv_z};
        const Vector3f& dv_c = world2cam(dv_w);
        const Vector2f& dv_uv = cam2pixel(dv_c);
        Eigen::Matrix2f affine = Eigen::Matrix2f::Identity();
        affine.col(0) = du_uv - uv;
        affine.col(1) = dv_uv - uv;
        float scale = std::pow(2, optical_layer_);
        int32_t shalf_patch_size = half_patch_size_ * scale;
        const ColorPatch& color_patch = feats->color_patches.at(feat_index);
        for (int32_t u = -shalf_patch_size * 2; u <= shalf_patch_size * 2; ++u) {
          for (int32_t v = -shalf_patch_size / 2; v <= shalf_patch_size / 2; ++v) {
            int32_t patch_index = (v + shalf_patch_size / 2) * (4 * shalf_patch_size + 1) +
                                  (u + shalf_patch_size * 2);
            Vector2f uv_patch{u, v};
            uv_patch = affine * uv_patch;
            Vector2f pixel_pos = uv + uv_patch;
            int32_t x = int32_t(pixel_pos.x() + 0.5f);
            int32_t y = int32_t(pixel_pos.y() + 0.5f);
            if (x >= 0 && x < patch_img.cols && y >= 0 && y < patch_img.rows) {
              uint8_t b_val = uint8_t(color_patch.at(0).at(patch_index));
              uint8_t g_val = uint8_t(color_patch.at(1).at(patch_index));
              uint8_t r_val = uint8_t(color_patch.at(2).at(patch_index));
              cv::Vec3b& pixel = patch_img.at<cv::Vec3b>(y, x);
              pixel[0] = b_val;
              pixel[1] = g_val;
              pixel[2] = r_val;
            }
          }
        }
        cv::circle(patch_img, cv::Point2f(uv.x(), uv.y()), 2, cv::Scalar(0, 255, 0), -1);
      }
    }
    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(cur_cam_header_, rgb_type, patch_img).toImageMsg();
    pub_patch_.publish(msg);
  }

  if (pub_rgbi_.getNumSubscribers() > 0) {
    cv::Mat intensity_img(height_, width_, CV_8UC1, cv::Scalar(0));
    se3::SE3 tfcw = tf_wc_.inverse();
    for (const HybridPoint& pt : scan_hybrid_->points) {
      const Eigen::Vector3f& pt_c = world2cam(pt.getVector3fMap());
      if (pt_c.z() <= kMathEpsilon) continue;
      const Eigen::Vector2f& uv = cam2pixel(pt_c);
      if (!InFov(uv, 1)) continue;
      cv::circle(intensity_img, cv::Point2f(uv.x(), uv.y()), 1,
                 cv::Scalar(pt.intensity + 1), -1);
    }
    cv::Mat pseudo_color, mask;
    cv::applyColorMap(intensity_img, pseudo_color, cv::COLORMAP_RAINBOW);
    cv::threshold(intensity_img, mask, 0, 255, cv::THRESH_BINARY_INV);
    pseudo_color.setTo(cv::Scalar(0, 0, 0), mask);
    cv::Mat rgbi_img = img_rgb_.clone();
    cv::add(pseudo_color, rgbi_img, rgbi_img);
    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(cur_cam_header_, rgb_type, rgbi_img).toImageMsg();
    pub_rgbi_.publish(msg);
  }

  if (pub_grad_.getNumSubscribers() > 0) {
    cv::normalize(img_grad_, img_grad_, 0, 255, cv::NORM_MINMAX);
    cv::Mat gradp_img;
    cv::Mat gradl_img;
    img_grad_.convertTo(gradp_img, CV_8UC1);
    img_grad_.convertTo(gradl_img, CV_8UC1);
    cv::cvtColor(gradp_img, gradp_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(gradl_img, gradl_img, cv::COLOR_GRAY2BGR);
    for (int32_t i = 0; i < pix_points_.size(); ++i) {
      const cv::Point2f& corner = pix_points_.at(i);
      cv::circle(gradp_img, corner, 4, cv::Scalar(0, 255, 0), -1);
    }
    for (int32_t i = 0; i < pix_lines_.size(); ++i) {
      const cv::Vec4f& line = pix_lines_.at(i);
      cv::line(gradl_img, cv::Point2f(line[0], line[1]),
               cv::Point2f(line[2], line[3]), cv::Scalar(0, 255, 0), 1);
    }
    for (int32_t i = 0; i < pix_samples_.size(); ++i) {
      const cv::Point2f& sample = pix_samples_.at(i);
      cv::circle(gradl_img, sample, 4, cv::Scalar(0, 255, 0), -1);
    }
    for (const auto& feat : vis_feats_->feats) {
      if (!feat.is_line) {
        const Eigen::Vector3f& pt_c = world2cam(feat.getVector3fMap());
        const Eigen::Vector2f& uv = cam2pixel(pt_c);
        cv::circle(gradp_img, cv::Point2f(uv.x(), uv.y()), 6, cv::Scalar(0, 0, 255), 2);
      } else {
        const Eigen::Vector3f& pt_c = world2cam(feat.getVector3fMap());
        const Eigen::Vector2f& uv = cam2pixel(pt_c);
        cv::circle(gradl_img, cv::Point2f(uv.x(), uv.y()), 6, cv::Scalar(0, 0, 255), 2);
      }
    }
    cv::Mat grad_img;
    cv::hconcat(gradp_img, gradl_img, grad_img);
    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(cur_cam_header_, rgb_type, grad_img).toImageMsg();
    pub_grad_.publish(msg);
  }

  if (pub_lines_.getNumSubscribers() > 0) {
    visualization_msgs::MarkerArray line_markers;

    visualization_msgs::Marker clear_marker;
    clear_marker.header = cur_cam_header_;
    clear_marker.header.frame_id = "camera_init";
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    line_markers.markers.push_back(clear_marker);

    int32_t marker_id = 0;

    for (int32_t i = 0; i < feat_lines_.size(); ++i) {
      const auto& line = feat_lines_.at(i);
      if (!line.is_valid) continue;

      visualization_msgs::Marker line_marker;
      line_marker.header = cur_cam_header_;
      line_marker.header.frame_id = "camera_init";
      line_marker.ns = "feat_lines";
      line_marker.id = marker_id++;
      line_marker.type = visualization_msgs::Marker::LINE_STRIP;
      line_marker.action = visualization_msgs::Marker::ADD;

      line_marker.pose.orientation.w = 1.0;
      line_marker.scale.x = 0.02;
      line_marker.color.r = 0.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 0.0;
      line_marker.color.a = 1.0;

      geometry_msgs::Point start_point, end_point;
      start_point.x = line.x;
      start_point.y = line.y;
      start_point.z = line.z;
      end_point.x = line.normal_x;
      end_point.y = line.normal_y;
      end_point.z = line.normal_z;

      line_marker.points.push_back(start_point);
      line_marker.points.push_back(end_point);
      line_markers.markers.push_back(line_marker);
    }

    for (int32_t i = 0; i < feat_samples_.size(); ++i) {
      const auto& sample = feat_samples_.at(i);
      if (!sample.is_line || !sample.is_valid) continue;

      visualization_msgs::Marker sample_marker;
      sample_marker.header = cur_cam_header_;
      sample_marker.header.frame_id = "camera_init";
      sample_marker.ns = "feat_samples";
      sample_marker.id = marker_id++;
      sample_marker.type = visualization_msgs::Marker::SPHERE;
      sample_marker.action = visualization_msgs::Marker::ADD;

      sample_marker.pose.position.x = sample.x;
      sample_marker.pose.position.y = sample.y;
      sample_marker.pose.position.z = sample.z;
      sample_marker.pose.orientation.w = 1.0;

      sample_marker.scale.x = sample_marker.scale.y = sample_marker.scale.z = 0.05;
      sample_marker.color.r = 1.0;
      sample_marker.color.g = 0.0;
      sample_marker.color.b = 0.0;
      sample_marker.color.a = 1.0;

      line_markers.markers.push_back(sample_marker);
    }

    pub_lines_.publish(line_markers);
  }

  if (pub_points_.getNumSubscribers() > 0) {
    visualization_msgs::MarkerArray point_markers;

    visualization_msgs::Marker clear_marker;
    clear_marker.header = cur_cam_header_;
    clear_marker.header.frame_id = "camera_init";
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    point_markers.markers.push_back(clear_marker);

    int32_t marker_id = 0;
    for (int32_t i = 0; i < grid_nums_; ++i) {
      if (grid_ptracked_.at(i)) {
        const VoxelVisualPtr& feats = grid_pfeat_.at(i);
        int32_t feat_index = pfeat_indice_.at(i);
        const auto& pt = feats->feats.at(feat_index);
        visualization_msgs::Marker marker;
        marker.header = cur_cam_header_;
        marker.header.frame_id = "camera_init";
        marker.ns = "tracked_points";
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = pt.x;
        marker.pose.position.y = pt.y;
        marker.pose.position.z = pt.z;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;
        point_markers.markers.push_back(marker);
      }
    }
    for (int32_t i = 0; i < feat_points_.size(); ++i) {
      const auto& pt = feat_points_.at(i);
      visualization_msgs::Marker marker;
      marker.header = cur_cam_header_;
      marker.header.frame_id = "camera_init";
      marker.ns = "feat_points";
      marker.id = marker_id++;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = pt.x;
      marker.pose.position.y = pt.y;
      marker.pose.position.z = pt.z;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = marker.scale.y = marker.scale.z = 0.08;
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      point_markers.markers.push_back(marker);
    }
    for (const auto& point : vis_feats_->feats) {
      if (!point.is_line) {
        visualization_msgs::Marker marker;
        marker.header = cur_cam_header_;
        marker.header.frame_id = "camera_init";
        marker.ns = "vis_points";
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = point.x;
        marker.pose.position.y = point.y;
        marker.pose.position.z = point.z;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.09;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        point_markers.markers.push_back(marker);
      }
    }
    pub_points_.publish(point_markers);
  }

  if (pub_cloud_.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*scan_color_, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(cur_cam_time_);
    cloud_msg.header.frame_id = "camera_init";
    pub_cloud_.publish(cloud_msg);
  }
}

void CamHandler::SavePcd() {
  if (pcd_save_en_) {
    std::string save_path = std::string(ROOT_DIR) + "Log/PCD/rgb_cloud.pcd";
    LOG(WARNING) << "saving pcd to " << save_path;
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(save_path, *scan_save_);
  }
}

}  // namespace livo

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
