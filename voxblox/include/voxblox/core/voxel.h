#ifndef VOXBLOX_CORE_VOXEL_H_
#define VOXBLOX_CORE_VOXEL_H_

#include <cstdint>
#include <string>

#include "voxblox/core/color.h"
#include "voxblox/core/common.h"

namespace voxblox {

struct TsdfSubVoxel {
  float distance = 0.0f;
  float weight = 0.0f;
  Color color;

  Semantics labels;
  SemanticProbabilities probabilities;

  // this is only used when usingBayesian update
  // TODO(jianhao): make this as parameter, num of class = 40 for now
  float rest_probabilities = std::log(1.0 / 40.0);
};

struct VoxelTypeStatus {
  bool is_this_type = false;
  bool updated = false;
  // to indicate if the eight bottom-left neighbors of the voxel is this type
  // e.g. If (neighbor_status & (1<<3)), then the fourth bottom-left neighbor is
  // this voxel type
  // uint8_t neighbor_status = 0;

  Eigen::Matrix<bool, 26, 1> neighbor_status;

  VoxelTypeStatus() {
    neighbor_status.setZero();  // Initialize all elements to false
  }

  // bool is_neighbor_of_this_type() const { return (neighbor_status != 0); }
  bool is_neighbor_of_this_type() const { return neighbor_status.any(); }
  bool should_be_divided() const {
    return (is_this_type || is_neighbor_of_this_type());
  }
};

struct TsdfVoxel {
  float distance = 0.0f;
  float weight = 0.0f;
  Color color;
  Semantics labels;
  SemanticProbabilities probabilities;

  // this is only used when usingBayesian update
  // TODO(jianhao): make this as parameter, num of class = 40 for now
  float rest_probabilities = std::log(1.0 / 40.0);

  // voxel status for subdivision due to small/middle semantics
  VoxelTypeStatus small_type_status;
  VoxelTypeStatus middle_type_status;

  // cannot use unique_ptr here because that won't allow to copy a layer
  // std::unique_ptr<voxblox::TsdfVoxel[]> child_voxels;

  std::vector<TsdfSubVoxel> child_voxels_queried;
  std::vector<TsdfSubVoxel> child_voxels_small;

  /********Variable used for CRF update****************/
  uint32_t current_label;
  std::vector<float> gradient;

  /********Variable used for geo complexity measure****************/
  float geo_weight = 0.0f;
  float geo_complexity = 0.0f;

  // indicate if it's semantic is updated after the latest CRF
  bool semantic_updated = false;
};

struct EsdfVoxel {
  float distance = 0.0f;
  float weight = 0.0f;

  bool observed = false;
  /**
   * Whether the voxel was copied from the TSDF (false) or created from a pose
   * or some other source (true). This member is not serialized!!!
   */
  bool hallucinated = false;
  bool in_queue = false;
  bool fixed = false;

  /**
   * Relative direction toward parent. If itself, then either uninitialized
   * or in the fixed frontier.
   */
  Eigen::Vector3i parent = Eigen::Vector3i::Zero();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct OccupancyVoxel {
  float probability_log = 0.0f;
  bool observed = false;
};

struct IntensityVoxel {
  float intensity = 0.0f;
  float weight = 0.0f;
};

/// Used for serialization only.
namespace voxel_types {
const std::string kNotSerializable = "not_serializable";
const std::string kTsdf = "tsdf";
const std::string kEsdf = "esdf";
const std::string kOccupancy = "occupancy";
const std::string kIntensity = "intensity";
}  // namespace voxel_types

template <typename Type>
std::string getVoxelType() {
  return voxel_types::kNotSerializable;
}

template <>
inline std::string getVoxelType<TsdfVoxel>() {
  return voxel_types::kTsdf;
}

template <>
inline std::string getVoxelType<EsdfVoxel>() {
  return voxel_types::kEsdf;
}

template <>
inline std::string getVoxelType<OccupancyVoxel>() {
  return voxel_types::kOccupancy;
}

template <>
inline std::string getVoxelType<IntensityVoxel>() {
  return voxel_types::kIntensity;
}

}  // namespace voxblox

#endif  // VOXBLOX_CORE_VOXEL_H_
