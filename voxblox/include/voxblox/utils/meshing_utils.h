#ifndef VOXBLOX_UTILS_MESHING_UTILS_H_
#define VOXBLOX_UTILS_MESHING_UTILS_H_

#include <voxblox/utils/color_maps.h>

#include "voxblox/core/common.h"
#include "voxblox/core/voxel.h"

namespace voxblox {

namespace utils {

template <typename VoxelType>
bool getSdfIfValid(const VoxelType& voxel, const FloatingPoint min_weight,
                   FloatingPoint* sdf);

template <>
inline bool getSdfIfValid(const TsdfSubVoxel& voxel,
                          const FloatingPoint min_weight, FloatingPoint* sdf) {
  DCHECK(sdf != nullptr);
  if (voxel.weight <= min_weight) {
    return false;
  }
  *sdf = voxel.distance;
  return true;
}

template <>
inline bool getSdfIfValid(const TsdfVoxel& voxel,
                          const FloatingPoint min_weight, FloatingPoint* sdf) {
  DCHECK(sdf != nullptr);
  if (voxel.weight <= min_weight) {
    return false;
  }
  *sdf = voxel.distance;
  return true;
}

template <>
inline bool getSdfIfValid(const EsdfVoxel& voxel,
                          const FloatingPoint /*min_weight*/,
                          FloatingPoint* sdf) {
  DCHECK(sdf != nullptr);
  if (!voxel.observed) {
    return false;
  }
  *sdf = voxel.distance;
  return true;
}

template <typename VoxelType>
bool getColorIfValid(const VoxelType& voxel, const FloatingPoint min_weight,
                     Color* color);

template <>
inline bool getColorIfValid(const TsdfSubVoxel& voxel,
                            const FloatingPoint min_weight, Color* color) {
  DCHECK(color != nullptr);
  if (voxel.weight <= min_weight) {
    return false;
  }
  *color = voxel.color;
  return true;
}

template <>
inline bool getColorIfValid(const TsdfVoxel& voxel,
                            const FloatingPoint min_weight, Color* color) {
  DCHECK(color != nullptr);
  if (voxel.weight <= min_weight) {
    return false;
  }
  *color = voxel.color;
  return true;
}

template <>
inline bool getColorIfValid(const EsdfVoxel& voxel,
                            const FloatingPoint /*min_weight*/, Color* color) {
  DCHECK(color != nullptr);
  if (!voxel.observed) {
    return false;
  }
  *color = Color(255u, 255u, 255u);
  return true;
}

template <typename VoxelType>
bool getColorIfValidSemantic(
    const VoxelType& voxel, const FloatingPoint min_weight, Color* color,
    const std::shared_ptr<voxblox::ColorMap>& color_map);

template <>
inline bool getColorIfValidSemantic(
    const TsdfVoxel& voxel, const FloatingPoint min_weight, Color* color,
    const std::shared_ptr<voxblox::ColorMap>& color_map) {
  DCHECK(color != nullptr);
  // // this is the code to show geometric complexity
  // std::vector<Color> viridis;

  // viridis.push_back(Color(68, 1, 84));
  // viridis.push_back(Color(70, 9, 92));
  // viridis.push_back(Color(71, 20, 102));
  // viridis.push_back(Color(72, 28, 110));
  // viridis.push_back(Color(71, 37, 117));
  // viridis.push_back(Color(71, 44, 123));
  // viridis.push_back(Color(69, 53, 128));
  // viridis.push_back(Color(67, 60, 132));
  // viridis.push_back(Color(64, 68, 135));
  // viridis.push_back(Color(61, 76, 137));
  // viridis.push_back(Color(58, 83, 139));
  // viridis.push_back(Color(54, 90, 140));
  // viridis.push_back(Color(51, 96, 141));
  // viridis.push_back(Color(48, 103, 141));
  // viridis.push_back(Color(46, 109, 142));
  // viridis.push_back(Color(43, 116, 142));
  // viridis.push_back(Color(40, 122, 142));
  // viridis.push_back(Color(38, 128, 142));
  // viridis.push_back(Color(35, 135, 141));
  // viridis.push_back(Color(33, 140, 141));
  // viridis.push_back(Color(31, 147, 139));
  // viridis.push_back(Color(30, 153, 138));
  // viridis.push_back(Color(30, 159, 136));
  // viridis.push_back(Color(32, 165, 133));
  // viridis.push_back(Color(37, 171, 129));
  // viridis.push_back(Color(44, 177, 125));
  // viridis.push_back(Color(53, 183, 120));
  // viridis.push_back(Color(64, 189, 114));
  // viridis.push_back(Color(75, 194, 108));
  // viridis.push_back(Color(89, 199, 100));
  // viridis.push_back(Color(103, 204, 92));
  // viridis.push_back(Color(119, 208, 82));
  // viridis.push_back(Color(136, 213, 71));
  // viridis.push_back(Color(151, 216, 62));
  // viridis.push_back(Color(170, 219, 50));
  // viridis.push_back(Color(186, 222, 39));
  // viridis.push_back(Color(205, 224, 29));
  // viridis.push_back(Color(220, 226, 24));
  // viridis.push_back(Color(238, 229, 27));
  // viridis.push_back(Color(253, 231, 36));
  // int index =
  //     static_cast<int>(voxel.geo_complexity * (viridis.size() - 1) / 0.1);
  // index = std::max(0, std::min(index, static_cast<int>(viridis.size()) - 1));
  // *color = viridis[index];
  // return true;

  // // this is the original code

  if (voxel.weight <= min_weight) {
    return false;
  }

  if (voxel.labels.size() == 0) {
    *color = Color(0u, 0u, 0u);
    return true;
  }
  SemanticProbabilities::const_iterator most_likely =
      std::max_element(voxel.probabilities.begin(), voxel.probabilities.end());

  *color = color_map->colorLookup(
      voxel.labels[std::distance(voxel.probabilities.begin(), most_likely)]);
  return true;
}

template <>
inline bool getColorIfValidSemantic(
    const EsdfVoxel& voxel, const FloatingPoint /*min_weight*/, Color* color,
    const std::shared_ptr<voxblox::ColorMap>& color_map) {
  DCHECK(color != nullptr);
  if (!voxel.observed) {
    return false;
  }
  *color = Color(255u, 255u, 255u);
  return true;
}

}  // namespace utils
}  // namespace voxblox

#endif  // VOXBLOX_UTILS_MESHING_UTILS_H_
