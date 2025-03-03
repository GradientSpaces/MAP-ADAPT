#include "voxblox/integrator/tsdf_integrator.h"

#include <float.h>

#include <iostream>
#include <list>

namespace voxblox {

TsdfIntegratorBase::Ptr TsdfIntegratorFactory::create(
    const std::string& integrator_type_name,
    const TsdfIntegratorBase::Config& config, Layer<TsdfVoxel>* layer) {
  CHECK(!integrator_type_name.empty());

  int integrator_type = 1;
  for (const std::string& valid_integrator_type_name :
       kTsdfIntegratorTypeNames) {
    if (integrator_type_name == valid_integrator_type_name) {
      return create(static_cast<TsdfIntegratorType>(integrator_type), config,
                    layer);
    }
    ++integrator_type;
  }
  LOG(FATAL) << "Unknown TSDF integrator type: " << integrator_type_name;
  return TsdfIntegratorBase::Ptr();
}

TsdfIntegratorBase::Ptr TsdfIntegratorFactory::create(
    const TsdfIntegratorType integrator_type,
    const TsdfIntegratorBase::Config& config, Layer<TsdfVoxel>* layer) {
  CHECK_NOTNULL(layer);
  switch (integrator_type) {
    case TsdfIntegratorType::kSimple:
      return TsdfIntegratorBase::Ptr(new SimpleTsdfIntegrator(config, layer));
      break;
    case TsdfIntegratorType::kMerged:
      return TsdfIntegratorBase::Ptr(new MergedTsdfIntegrator(config, layer));
      break;
    case TsdfIntegratorType::kFast:
      return TsdfIntegratorBase::Ptr(new FastTsdfIntegrator(config, layer));
      break;
    default:
      LOG(FATAL) << "Unknown TSDF integrator type: "
                 << static_cast<int>(integrator_type);
      break;
  }
  return TsdfIntegratorBase::Ptr();
}

// Note many functions state if they are thread safe. Unless explicitly stated
// otherwise, this thread safety is based on the assumption that any pointers
// passed to the functions point to objects that are guaranteed to not be
// accessed by other threads.

TsdfIntegratorBase::TsdfIntegratorBase(const Config& config,
                                       Layer<TsdfVoxel>* layer)
    : config_(config) {
  setLayer(layer);

  if (config_.integrator_threads == 0) {
    LOG(WARNING) << "Automatic core count failed, defaulting to 1 threads";
    config_.integrator_threads = 1;
  }
  // clearing rays have no utility if voxel_carving is disabled
  if (config_.allow_clear && !config_.voxel_carving_enabled) {
    config_.allow_clear = false;
  }
}

void TsdfIntegratorBase::setLayer(Layer<TsdfVoxel>* layer) {
  CHECK_NOTNULL(layer);

  layer_ = layer;

  voxel_size_ = layer_->voxel_size();
  block_size_ = layer_->block_size();
  voxels_per_side_ = layer_->voxels_per_side();

  voxel_size_inv_ = 1.0 / voxel_size_;
  block_size_inv_ = 1.0 / block_size_;
  voxels_per_side_inv_ = 1.0 / voxels_per_side_;
}

// Will return a pointer to a voxel located at global_voxel_idx in the tsdf
// layer. Thread safe.
// Takes in the last_block_idx and last_block to prevent unneeded map lookups.
// If the block this voxel would be in has not been allocated, a block in
// temp_block_map_ is created/accessed and a voxel from this map is returned
// instead. Unlike the layer, accessing temp_block_map_ is controlled via a
// mutex allowing it to grow during integration.
// These temporary blocks can be merged into the layer later by calling
// updateLayerWithStoredBlocks()
TsdfVoxel* TsdfIntegratorBase::allocateStorageAndGetVoxelPtr(
    const GlobalIndex& global_voxel_idx, Block<TsdfVoxel>::Ptr* last_block,
    BlockIndex* last_block_idx) {
  DCHECK(last_block != nullptr);
  DCHECK(last_block_idx != nullptr);

  const BlockIndex block_idx =
      getBlockIndexFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_inv_);

  if ((block_idx != *last_block_idx) || (*last_block == nullptr)) {
    *last_block = layer_->getBlockPtrByIndex(block_idx);
    *last_block_idx = block_idx;
  }

  // If no block at this location currently exists, we allocate a temporary
  // voxel that will be merged into the map later
  if (*last_block == nullptr) {
    // To allow temp_block_map_ to grow we can only let one thread in at once
    std::lock_guard<std::mutex> lock(temp_block_mutex_);

    typename Layer<TsdfVoxel>::BlockHashMap::iterator it =
        temp_block_map_.find(block_idx);
    if (it != temp_block_map_.end()) {
      *last_block = it->second;
    } else {
      auto insert_status = temp_block_map_.emplace(
          block_idx, std::make_shared<Block<TsdfVoxel>>(
                         voxels_per_side_, voxel_size_,
                         getOriginPointFromGridIndex(block_idx, block_size_)));

      DCHECK(insert_status.second) << "Block already exists when allocating at "
                                   << block_idx.transpose();

      *last_block = insert_status.first->second;
    }
  }

  (*last_block)->updated().set();

  const VoxelIndex local_voxel_idx =
      getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_);

  return &((*last_block)->getVoxelByVoxelIndex(local_voxel_idx));
}

// NOT thread safe
void TsdfIntegratorBase::updateLayerWithStoredBlocks() {
  BlockIndex last_block_idx;
  Block<TsdfVoxel>::Ptr block = nullptr;

  for (const std::pair<const BlockIndex, Block<TsdfVoxel>::Ptr>&
           temp_block_pair : temp_block_map_) {
    layer_->insertBlock(temp_block_pair);
  }

  temp_block_map_.clear();
}

TsdfVoxel TsdfIntegratorBase::getNeighborVoxel(
    VoxelIndex& neighbor_voxel_index, Block<TsdfVoxel>::Ptr& current_block,
    bool* success) {
  if (current_block->isValidVoxelIndex(neighbor_voxel_index)) {
    *success = true;
    return current_block->getVoxelByVoxelIndex(neighbor_voxel_index);
  } else {
    BlockIndex block_offset = BlockIndex::Zero();

    for (unsigned int j = 0u; j < 3u; j++) {
      if (neighbor_voxel_index(j) < 0) {
        block_offset(j) = -1;
        neighbor_voxel_index(j) = neighbor_voxel_index(j) + voxels_per_side_;
      } else if (neighbor_voxel_index(j) >=
                 static_cast<IndexElement>(voxels_per_side_)) {
        block_offset(j) = 1;
        neighbor_voxel_index(j) = neighbor_voxel_index(j) - voxels_per_side_;
      }
    }

    BlockIndex neighbor_block_index =
        current_block->block_index() + block_offset;
    if (layer_->hasBlock(neighbor_block_index)) {
      Block<TsdfVoxel>::Ptr neighbor_block =
          layer_->getBlockPtrByIndex(neighbor_block_index);
      *success = true;
      return neighbor_block->getVoxelByVoxelIndex(neighbor_voxel_index);
    } else {
      // this neighbor voxel doesn't exist
      TsdfVoxel output;
      *success = false;
      return output;
    }
  }
}

TsdfVoxel* TsdfIntegratorBase::getNeighborVoxelPtr(
    VoxelIndex& neighbor_voxel_index, Block<TsdfVoxel>::Ptr& current_block,
    bool* success) {
  if (current_block->isValidVoxelIndex(neighbor_voxel_index)) {
    *success = true;
    return &(current_block->getVoxelByVoxelIndex(neighbor_voxel_index));
  } else {
    BlockIndex block_offset = BlockIndex::Zero();

    for (unsigned int j = 0u; j < 3u; j++) {
      if (neighbor_voxel_index(j) < 0) {
        block_offset(j) = -1;
        neighbor_voxel_index(j) = neighbor_voxel_index(j) + voxels_per_side_;
      } else if (neighbor_voxel_index(j) >=
                 static_cast<IndexElement>(voxels_per_side_)) {
        block_offset(j) = 1;
        neighbor_voxel_index(j) = neighbor_voxel_index(j) - voxels_per_side_;
      }
    }

    BlockIndex neighbor_block_index =
        current_block->block_index() + block_offset;
    if (layer_->hasBlock(neighbor_block_index)) {
      Block<TsdfVoxel>::Ptr neighbor_block =
          layer_->getBlockPtrByIndex(neighbor_block_index);
      *success = true;
      return &(neighbor_block->getVoxelByVoxelIndex(neighbor_voxel_index));
    } else {
      // this neighbor voxel doesn't exist
      TsdfVoxel output;
      *success = false;
      return &(output);
    }
  }
}

// Updates tsdf_voxel. Thread safe.
void TsdfIntegratorBase::updateTsdfVoxel(const Point& origin,
                                         const Point& point_G,
                                         const GlobalIndex& global_voxel_idx,
                                         const Color& color, const float weight,
                                         TsdfVoxel* tsdf_voxel) {
  DCHECK(tsdf_voxel != nullptr);

  const Point voxel_center =
      getCenterPointFromGridIndex(global_voxel_idx, voxel_size_);

  const float sdf = computeDistance(origin, point_G, voxel_center);

  float updated_weight = weight;
  // Compute updated weight in case we use weight dropoff. It's
  // easier here that in getVoxelWeight as here we have the actual
  // SDF for the voxel already computed.
  const FloatingPoint dropoff_epsilon = voxel_size_;
  if (config_.use_weight_dropoff && sdf < -dropoff_epsilon) {
    updated_weight = weight * (config_.default_truncation_distance + sdf) /
                     (config_.default_truncation_distance - dropoff_epsilon);
    updated_weight = std::max(updated_weight, 0.0f);
  }

  // Compute the updated weight in case we compensate for
  // sparsity. By multiplicating the weight of occupied areas
  // (|sdf| < truncation distance) by a factor, we prevent to
  // easily fade out these areas with the free space parts of
  // other rays which pass through the corresponding voxels. This
  // can be useful for creating a TSDF map from sparse sensor data
  // (e.g. visual features from a SLAM system). By default, this
  // option is disabled.
  if (config_.use_sparsity_compensation_factor) {
    if (std::abs(sdf) < config_.default_truncation_distance) {
      updated_weight *= config_.sparsity_compensation_factor;
    }
  }

  // Lookup the mutex that is responsible for this voxel and lock
  // it
  std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

  const float new_weight = tsdf_voxel->weight + updated_weight;

  // it is possible to have weights very close to zero, due to the
  // limited precision of floating points dividing by this small
  // value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }

  const float new_sdf =
      (sdf * updated_weight + tsdf_voxel->distance * tsdf_voxel->weight) /
      new_weight;

  // color blending is expensive only do it close to the surface
  if (std::abs(sdf) < config_.default_truncation_distance) {
    tsdf_voxel->color = Color::blendTwoColors(
        tsdf_voxel->color, tsdf_voxel->weight, color, updated_weight);
  }
  tsdf_voxel->distance =
      (new_sdf > 0.0) ? std::min(config_.default_truncation_distance, new_sdf)
                      : std::max(-config_.default_truncation_distance, new_sdf);
  tsdf_voxel->weight = std::min(config_.max_weight, new_weight);
}

void TsdfIntegratorBase::updateTsdfVoxelSemanticProbability(
    const Point& origin, const Point& point_G,
    const GlobalIndex& global_voxel_idx, const Color& color,
    const uint32_t& semantic_top4_encoded,
    const uint32_t& probabilities_top4_encoded, const float weight,
    TsdfVoxel* tsdf_voxel) {
  DCHECK(tsdf_voxel != nullptr);

  const Point voxel_center =
      getCenterPointFromGridIndex(global_voxel_idx, voxel_size_);

  const float sdf = computeDistance(origin, point_G, voxel_center);

  float updated_weight = weight;
  // Compute updated weight in case we use weight dropoff. It's
  // easier here that in getVoxelWeight as here we have the actual
  // SDF for the voxel already computed.
  const FloatingPoint dropoff_epsilon = voxel_size_;
  if (config_.use_weight_dropoff && sdf < -dropoff_epsilon) {
    updated_weight = weight * (config_.default_truncation_distance + sdf) /
                     (config_.default_truncation_distance - dropoff_epsilon);
    updated_weight = std::max(updated_weight, 0.0f);
  }

  // Compute the updated weight in case we compensate for
  // sparsity. By multiplicating the weight of occupied areas
  // (|sdf| < truncation distance) by a factor, we prevent to
  // easily fade out these areas with the free space parts of
  // other rays which pass through the corresponding voxels. This
  // can be useful for creating a TSDF map from sparse sensor data
  // (e.g. visual features from a SLAM system). By default, this
  // option is disabled.
  if (config_.use_sparsity_compensation_factor) {
    if (std::abs(sdf) < config_.default_truncation_distance) {
      updated_weight *= config_.sparsity_compensation_factor;
    }
  }

  // Lookup the mutex that is responsible for this voxel and lock
  // it
  std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

  const float new_weight = tsdf_voxel->weight + updated_weight;

  // it is possible to have weights very close to zero, due to the
  // limited precision of floating points dividing by this small
  // value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }

  const float new_sdf =
      (sdf * updated_weight + tsdf_voxel->distance * tsdf_voxel->weight) /
      new_weight;

  // color blending is expensive only do it close to the surface
  if (std::abs(sdf) < config_.default_truncation_distance) {
    tsdf_voxel->color = Color::blendTwoColors(
        tsdf_voxel->color, tsdf_voxel->weight, color, updated_weight);
  }

  // TODO(jianhao) This threshold needs to be fintuned
  // if (std::abs(sdf) < voxel_size_ * std::sqrt(3))
  // if (abs(point_G.x() - voxel_center.x()) < voxel_size_ / 1.0
  // &&
  //     abs(point_G.y() - voxel_center.y()) < voxel_size_ / 1.0
  //     && abs(point_G.z() - voxel_center.z()) < voxel_size_
  //     / 1.0)
  if (std::abs(sdf) < voxel_size_) {
    float semantic_updated_weight = updated_weight;
    if (abs(point_G.x() - voxel_center.x()) > voxel_size_ / 2.0 ||
        abs(point_G.y() - voxel_center.y()) > voxel_size_ / 2.0 ||
        abs(point_G.z() - voxel_center.z()) > voxel_size_ / 2.0) {
      const float dist_G_voxel = (point_G - voxel_center).norm();
      semantic_updated_weight *= (voxel_size_ / 2.0) / dist_G_voxel;
      // semantic_updated_weight *= (voxel_size_ / 2.0) * (voxel_size_ / 2.0)
      // /
      //                            (dist_G_voxel * dist_G_voxel);
    }
    fuseSemanticProbability(tsdf_voxel, semantic_top4_encoded,
                            probabilities_top4_encoded,
                            semantic_updated_weight);

    if (adaptive_mapping) {
      if (tsdf_voxel->labels.size() > 0) {
        // expand the voxel if this voxel turns out to be queried
        // class
        SemanticProbabilities::iterator most_likely = std::max_element(
            tsdf_voxel->probabilities.begin(), tsdf_voxel->probabilities.end());
        uint32_t most_likely_class = tsdf_voxel->labels[std::distance(
            tsdf_voxel->probabilities.begin(), most_likely)];

        // 1 is queried, 2 is middle, 3 is large
        uint8_t which_type = 0;
        if (std::count(small_semantic.begin(), small_semantic.end(),
                       most_likely_class)) {
          which_type = 1;
        } else if (std::count(large_semantic.begin(), large_semantic.end(),
                              most_likely_class)) {
          which_type = 3;
        } else {
          which_type = 2;
        }

        if (which_type == 1) {
          // this is a queried semantic voxel
          if (tsdf_voxel->child_voxels_queried.size() !=
              std::pow(adaptive_ratio_small, 3)) {
            CHECK(!(tsdf_voxel->small_type_status.should_be_divided()));
            std::vector<TsdfSubVoxel>().swap(tsdf_voxel->child_voxels_queried);
            tsdf_voxel->child_voxels_queried.resize(adaptive_ratio_small *
                                                    adaptive_ratio_small *
                                                    adaptive_ratio_small);
          }

          if (!(tsdf_voxel->small_type_status.is_this_type)) {
            tsdf_voxel->small_type_status.is_this_type = true;
            tsdf_voxel->small_type_status.updated = true;
          }

          // we don't remove middle size subvoxels in this case, as it should
          // have more geo info when it's queried semantic
        } else if (which_type == 2) {
          // this is a middle semantic voxel
          if (tsdf_voxel->child_voxels_small.size() !=
              std::pow(adaptive_ratio_middle, 3)) {
            CHECK(!(tsdf_voxel->middle_type_status.should_be_divided()));
            std::vector<TsdfSubVoxel>().swap(tsdf_voxel->child_voxels_small);
            tsdf_voxel->child_voxels_small.resize(adaptive_ratio_middle *
                                                  adaptive_ratio_middle *
                                                  adaptive_ratio_middle);
          }

          if (!(tsdf_voxel->middle_type_status.is_this_type)) {
            tsdf_voxel->middle_type_status.is_this_type = true;
            tsdf_voxel->middle_type_status.updated = true;
          }

          // check if we need to erase queried semantic subvoxels
          if (tsdf_voxel->small_type_status.is_this_type) {
            FloatingPoint max_log_prob = *most_likely;
            FloatingPoint sum = 0.0;
            for (FloatingPoint log_prob : tsdf_voxel->probabilities) {
              sum += std::exp(log_prob - max_log_prob);
            }
            // TODO(jianhao) this 40 must be a parameter (lenght of nyu40
            // semantics)
            sum +=
                (40.0 - static_cast<float>(tsdf_voxel->probabilities.size())) *
                std::exp(tsdf_voxel->rest_probabilities - max_log_prob);
            FloatingPoint max_prob = 1.0 / sum;

            // we are pretty sure it's a middle semantic, earse queried
            // subvoxels
            if (max_prob > 0.95) {
              // it's no longer a small one
              tsdf_voxel->small_type_status.is_this_type = false;
              // its type updated
              tsdf_voxel->small_type_status.updated = true;
              if (!(tsdf_voxel->small_type_status.is_neighbor_of_this_type())) {
                // it's not a neighbor of small semantic, earse subvoxels
                std::vector<TsdfSubVoxel>().swap(
                    tsdf_voxel->child_voxels_queried);
              }
            }
          }
        } else if ((tsdf_voxel->small_type_status.is_this_type) ||
                   (tsdf_voxel->middle_type_status.is_this_type)) {
          // it's large semantic, check if we need to earse subvoxels
          FloatingPoint max_log_prob = *most_likely;
          FloatingPoint sum = 0.0;
          for (FloatingPoint log_prob : tsdf_voxel->probabilities) {
            sum += std::exp(log_prob - max_log_prob);
          }
          // TODO(jianhao) this 40 must be a parameter (lenght of nyu40
          // semantics)
          sum += (40.0 - static_cast<float>(tsdf_voxel->probabilities.size())) *
                 std::exp(tsdf_voxel->rest_probabilities - max_log_prob);
          FloatingPoint max_prob = 1.0 / sum;

          // we are pretty sure it's a large semantic, earse subvoxels
          if (max_prob > 0.95) {
            // process queried
            if (tsdf_voxel->small_type_status.is_this_type) {
              // it's no longer a small one
              tsdf_voxel->small_type_status.is_this_type = false;
              // its type updated
              tsdf_voxel->small_type_status.updated = true;
              if (!(tsdf_voxel->small_type_status.is_neighbor_of_this_type())) {
                // it's not a neighbor of small semantic, earse subvoxels
                std::vector<TsdfSubVoxel>().swap(
                    tsdf_voxel->child_voxels_queried);
              }
            }

            // process middle
            if (tsdf_voxel->middle_type_status.is_this_type) {
              // it's no longer a middle one
              tsdf_voxel->middle_type_status.is_this_type = false;
              // its type updated
              tsdf_voxel->middle_type_status.updated = true;
              if (!(tsdf_voxel->middle_type_status
                        .is_neighbor_of_this_type())) {
                // it's not a neighbor of middle semantic, earse subvoxels
                std::vector<TsdfSubVoxel>().swap(
                    tsdf_voxel->child_voxels_small);
              }
            }
          }
        }
      }
    }
  }

  tsdf_voxel->distance =
      (new_sdf > 0.0) ? std::min(config_.default_truncation_distance, new_sdf)
                      : std::max(-config_.default_truncation_distance, new_sdf);
  tsdf_voxel->weight = std::min(config_.max_weight, new_weight);
}

void TsdfIntegratorBase::updateTsdfSubVoxelSemanticProbability(
    const Point& origin, const Point& point_G, const Point voxel_center,
    const Color& color, const uint32_t& semantic_top4_encoded,
    const uint32_t& probabilities_top4_encoded, const float weight,
    TsdfSubVoxel* tsdf_voxel, int voxel_size_ratio) {
  DCHECK(tsdf_voxel != nullptr);

  FloatingPoint child_voxel_size =
      voxel_size_ / static_cast<FloatingPoint>(voxel_size_ratio);
  FloatingPoint sub_level_truncation_distance =
      config_.default_truncation_distance /
      static_cast<FloatingPoint>(voxel_size_ratio);

  const float sdf = computeDistance(origin, point_G, voxel_center);

  // TODO(jianhao): maybe there's a better way to filter this
  // if (sdf > voxel_size_ /
  // static_cast<FloatingPoint>(voxel_size_ratio))
  // {
  //   return;
  // }

  float updated_weight = weight;

  const FloatingPoint dropoff_epsilon = child_voxel_size;
  if (config_.use_weight_dropoff && sdf < -dropoff_epsilon) {
    updated_weight = weight * (sub_level_truncation_distance + sdf) /
                     (sub_level_truncation_distance - dropoff_epsilon);
    updated_weight = std::max(updated_weight, 0.0f);
  }

  if (config_.use_sparsity_compensation_factor) {
    if (std::abs(sdf) < sub_level_truncation_distance) {
      updated_weight *= config_.sparsity_compensation_factor;
    }
  }

  const float new_weight = tsdf_voxel->weight + updated_weight;

  // it is possible to have weights very close to zero, due to the
  // limited precision of floating points dividing by this small
  // value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }

  const float new_sdf =
      (sdf * updated_weight + tsdf_voxel->distance * tsdf_voxel->weight) /
      new_weight;

  if (std::abs(sdf) < sub_level_truncation_distance) {
    tsdf_voxel->color = Color::blendTwoColors(
        tsdf_voxel->color, tsdf_voxel->weight, color, updated_weight);
  }

  /* For sub voxel, we fuse the semantic label only when required*/
  if (fuse_semantic_for_subvoxel) {
    if (std::abs(sdf) < child_voxel_size) {
      float semantic_updated_weight = updated_weight;
      if (abs(point_G.x() - voxel_center.x()) > child_voxel_size / 2.0 ||
          abs(point_G.y() - voxel_center.y()) > child_voxel_size / 2.0 ||
          abs(point_G.z() - voxel_center.z()) > child_voxel_size / 2.0) {
        const float dist_G_voxel = (point_G - voxel_center).norm();
        semantic_updated_weight *= (child_voxel_size / 2.0) / dist_G_voxel;
        // semantic_updated_weight *= (voxel_size_ / 2.0) * (voxel_size_
        // / 2.0)
        // /
        //                            (dist_G_voxel * dist_G_voxel);
      }
      fuseSemanticProbability(tsdf_voxel, semantic_top4_encoded,
                              probabilities_top4_encoded,
                              semantic_updated_weight);
    }
  }

  tsdf_voxel->distance =
      (new_sdf > 0.0) ? std::min(sub_level_truncation_distance, new_sdf)
                      : std::max(-sub_level_truncation_distance, new_sdf);
  tsdf_voxel->weight = std::min(config_.max_weight, new_weight);
}

void TsdfIntegratorBase::updateTsdfVoxelGeoComplexity(
    const Point& origin, const Point& point_G,
    const GlobalIndex& global_voxel_idx, const Color& color,
    const float& geo_complexity, const float weight, TsdfVoxel* tsdf_voxel) {
  DCHECK(tsdf_voxel != nullptr);

  const Point voxel_center =
      getCenterPointFromGridIndex(global_voxel_idx, voxel_size_);

  const float sdf = computeDistance(origin, point_G, voxel_center);

  float updated_weight = weight;
  // Compute updated weight in case we use weight dropoff. It's
  // easier here that in getVoxelWeight as here we have the actual
  // SDF for the voxel already computed.
  const FloatingPoint dropoff_epsilon = voxel_size_;
  if (config_.use_weight_dropoff && sdf < -dropoff_epsilon) {
    updated_weight = weight * (config_.default_truncation_distance + sdf) /
                     (config_.default_truncation_distance - dropoff_epsilon);
    updated_weight = std::max(updated_weight, 0.0f);
  }

  // Compute the updated weight in case we compensate for
  // sparsity. By multiplicating the weight of occupied areas
  // (|sdf| < truncation distance) by a factor, we prevent to
  // easily fade out these areas with the free space parts of
  // other rays which pass through the corresponding voxels. This
  // can be useful for creating a TSDF map from sparse sensor data
  // (e.g. visual features from a SLAM system). By default, this
  // option is disabled.
  if (config_.use_sparsity_compensation_factor) {
    if (std::abs(sdf) < config_.default_truncation_distance) {
      updated_weight *= config_.sparsity_compensation_factor;
    }
  }

  // Lookup the mutex that is responsible for this voxel and lock
  // it
  std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

  const float new_weight = tsdf_voxel->weight + updated_weight;

  // it is possible to have weights very close to zero, due to the
  // limited precision of floating points dividing by this small
  // value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }

  const float new_sdf =
      (sdf * updated_weight + tsdf_voxel->distance * tsdf_voxel->weight) /
      new_weight;

  // color blending is expensive only do it close to the surface
  if (std::abs(sdf) < config_.default_truncation_distance) {
    tsdf_voxel->color = Color::blendTwoColors(
        tsdf_voxel->color, tsdf_voxel->weight, color, updated_weight);
  }

  if (std::abs(sdf) < voxel_size_) {
    if (geo_complexity > 0) {
      const float geo_new_weight = tsdf_voxel->geo_weight + updated_weight;
      tsdf_voxel->geo_complexity =
          (geo_complexity * updated_weight +
           tsdf_voxel->geo_complexity * tsdf_voxel->geo_weight) /
          geo_new_weight;
      tsdf_voxel->geo_weight = std::min(config_.max_weight, geo_new_weight);
    }
  }

  tsdf_voxel->distance =
      (new_sdf > 0.0) ? std::min(config_.default_truncation_distance, new_sdf)
                      : std::max(-config_.default_truncation_distance, new_sdf);
  tsdf_voxel->weight = std::min(config_.max_weight, new_weight);
}

void TsdfIntegratorBase::updateTsdfVoxelGeoSemantic(
    const Point& origin, const Point& point_G,
    const GlobalIndex& global_voxel_idx, const Color& color,
    const float& geo_complexity, const uint32_t& semantic_top4_encoded,
    const uint32_t& probabilities_top4_encoded, const float weight,
    TsdfVoxel* tsdf_voxel) {
  DCHECK(tsdf_voxel != nullptr);

  const Point voxel_center =
      getCenterPointFromGridIndex(global_voxel_idx, voxel_size_);

  const float sdf = computeDistance(origin, point_G, voxel_center);

  float updated_weight = weight;
  // Compute updated weight in case we use weight dropoff. It's
  // easier here that in getVoxelWeight as here we have the actual
  // SDF for the voxel already computed.
  const FloatingPoint dropoff_epsilon = voxel_size_;
  if (config_.use_weight_dropoff && sdf < -dropoff_epsilon) {
    updated_weight = weight * (config_.default_truncation_distance + sdf) /
                     (config_.default_truncation_distance - dropoff_epsilon);
    updated_weight = std::max(updated_weight, 0.0f);
  }

  // Compute the updated weight in case we compensate for
  // sparsity. By multiplicating the weight of occupied areas
  // (|sdf| < truncation distance) by a factor, we prevent to
  // easily fade out these areas with the free space parts of
  // other rays which pass through the corresponding voxels. This
  // can be useful for creating a TSDF map from sparse sensor data
  // (e.g. visual features from a SLAM system). By default, this
  // option is disabled.
  if (config_.use_sparsity_compensation_factor) {
    if (std::abs(sdf) < config_.default_truncation_distance) {
      updated_weight *= config_.sparsity_compensation_factor;
    }
  }

  // Lookup the mutex that is responsible for this voxel and lock
  // it
  std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

  const float new_weight = tsdf_voxel->weight + updated_weight;

  // it is possible to have weights very close to zero, due to the
  // limited precision of floating points dividing by this small
  // value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }

  const float new_sdf =
      (sdf * updated_weight + tsdf_voxel->distance * tsdf_voxel->weight) /
      new_weight;

  // color blending is expensive only do it close to the surface
  if (std::abs(sdf) < config_.default_truncation_distance) {
    tsdf_voxel->color = Color::blendTwoColors(
        tsdf_voxel->color, tsdf_voxel->weight, color, updated_weight);
  }

  // TODO(jianhao) This threshold needs to be fintuned
  // if (std::abs(sdf) < voxel_size_ * std::sqrt(3))
  // if (abs(point_G.x() - voxel_center.x()) < voxel_size_ / 1.0
  // &&
  //     abs(point_G.y() - voxel_center.y()) < voxel_size_ / 1.0
  //     && abs(point_G.z() - voxel_center.z()) < voxel_size_
  //     / 1.0)
  if (std::abs(sdf) < voxel_size_) {
    float new_updated_weight = updated_weight;
    if (abs(point_G.x() - voxel_center.x()) > voxel_size_ / 2.0 ||
        abs(point_G.y() - voxel_center.y()) > voxel_size_ / 2.0 ||
        abs(point_G.z() - voxel_center.z()) > voxel_size_ / 2.0) {
      const float dist_G_voxel = (point_G - voxel_center).norm();
      new_updated_weight *= (voxel_size_ / 2.0) / dist_G_voxel;
      // semantic_updated_weight *= (voxel_size_ / 2.0) * (voxel_size_ / 2.0)
      // /
      //                            (dist_G_voxel * dist_G_voxel);
    }
    /************************Update geo complexity***********************/
    // the input geo_complexity could be -1, which means we didn't measure the
    // geo complexity on this pixel
    if (geo_complexity > 0.0f) {
      const float geo_new_weight = tsdf_voxel->geo_weight + new_updated_weight;
      tsdf_voxel->geo_complexity =
          (geo_complexity * new_updated_weight +
           tsdf_voxel->geo_complexity * tsdf_voxel->geo_weight) /
          geo_new_weight;
      tsdf_voxel->geo_weight = std::min(config_.max_weight, geo_new_weight);
    }

    /************************Update semantics***********************/
    fuseSemanticProbability(tsdf_voxel, semantic_top4_encoded,
                            probabilities_top4_encoded, new_updated_weight);

    if (adaptive_mapping) {
      // 1 is small, 2 is middle, 3 is large
      uint8_t which_type = 0;
      SemanticProbabilities::iterator most_likely;
      if (tsdf_voxel->labels.size() > 0) {
        most_likely = std::max_element(tsdf_voxel->probabilities.begin(),
                                       tsdf_voxel->probabilities.end());
        uint32_t most_likely_class = tsdf_voxel->labels[std::distance(
            tsdf_voxel->probabilities.begin(), most_likely)];

        if (std::count(small_semantic.begin(), small_semantic.end(),
                       most_likely_class)) {
          which_type = 1;
        } else if (std::count(large_semantic.begin(), large_semantic.end(),
                              most_likely_class)) {
          which_type = 3;
        } else {
          which_type = 2;
        }
      } else {
        which_type = 3;
      }

      if (tsdf_voxel->geo_complexity < geo_thresholds[0]) {
        which_type = std::min(which_type, static_cast<uint8_t>(3));
      } else if (tsdf_voxel->geo_complexity < geo_thresholds[1]) {
        which_type = std::min(which_type, static_cast<uint8_t>(2));
      } else {
        which_type = 1;
      }

      if (which_type == 1) {
        // this is a queried semantic voxel
        if (tsdf_voxel->child_voxels_queried.size() !=
            std::pow(adaptive_ratio_small, 3)) {
          CHECK(!(tsdf_voxel->small_type_status.should_be_divided()));
          std::vector<TsdfSubVoxel>().swap(tsdf_voxel->child_voxels_queried);
          tsdf_voxel->child_voxels_queried.resize(adaptive_ratio_small *
                                                  adaptive_ratio_small *
                                                  adaptive_ratio_small);
        }

        if (!(tsdf_voxel->small_type_status.is_this_type)) {
          tsdf_voxel->small_type_status.is_this_type = true;
          tsdf_voxel->small_type_status.updated = true;
        }

        // we don't remove middle size subvoxels in this case, as it should
        // have more geo info when it's queried semantic
      } else if (which_type == 2) {
        // this is a middle semantic voxel
        if (tsdf_voxel->child_voxels_small.size() !=
            std::pow(adaptive_ratio_middle, 3)) {
          CHECK(!(tsdf_voxel->middle_type_status.should_be_divided()));
          std::vector<TsdfSubVoxel>().swap(tsdf_voxel->child_voxels_small);
          tsdf_voxel->child_voxels_small.resize(adaptive_ratio_middle *
                                                adaptive_ratio_middle *
                                                adaptive_ratio_middle);
        }

        if (!(tsdf_voxel->middle_type_status.is_this_type)) {
          tsdf_voxel->middle_type_status.is_this_type = true;
          tsdf_voxel->middle_type_status.updated = true;
        }

        // check if we need to erase queried semantic subvoxels
        if ((tsdf_voxel->small_type_status.is_this_type)) {
          // a bit looser geometric threshold here to avoid mistakenly remove
          // rich information
          if ((tsdf_voxel->geo_complexity <
               0.2f * geo_thresholds[0] + 0.8f * geo_thresholds[1])) {
            // we also need to make sure it's semantic is confident
            FloatingPoint max_prob;
            if (tsdf_voxel->labels.size() == 0) {
              // we have no semantic information integrated
              max_prob = 1.0f;
            } else {
              FloatingPoint max_log_prob = *most_likely;
              FloatingPoint sum = 0.0;
              for (FloatingPoint log_prob : tsdf_voxel->probabilities) {
                sum += std::exp(log_prob - max_log_prob);
              }
              // TODO(jianhao) this 40 must be a parameter (lenght of nyu40
              // semantics)
              sum += (40.0 -
                      static_cast<float>(tsdf_voxel->probabilities.size())) *
                     std::exp(tsdf_voxel->rest_probabilities - max_log_prob);
              max_prob = 1.0 / sum;
            }

            // we are pretty confident on the semantic estimation, earse queried
            // subvoxels
            if (max_prob > 0.95) {
              //  it's no longer a small one
              tsdf_voxel->small_type_status.is_this_type = false;
              // its type updated
              tsdf_voxel->small_type_status.updated = true;
              if (!(tsdf_voxel->small_type_status.is_neighbor_of_this_type())) {
                // it's not a neighbor of small semantic, earse subvoxels
                std::vector<TsdfSubVoxel>().swap(
                    tsdf_voxel->child_voxels_queried);
              }
            }
          }
        }
      } else if ((tsdf_voxel->small_type_status.is_this_type) ||
                 (tsdf_voxel->middle_type_status.is_this_type)) {
        // it's large semantic, check if we need to earse subvoxels

        // check semantic confidence
        FloatingPoint max_prob;
        if (tsdf_voxel->labels.size() == 0) {
          // we have no semantic information integrated
          max_prob = 1.0f;
        } else {
          FloatingPoint max_log_prob = *most_likely;
          FloatingPoint sum = 0.0;
          for (FloatingPoint log_prob : tsdf_voxel->probabilities) {
            sum += std::exp(log_prob - max_log_prob);
          }
          // TODO(jianhao) this 40 must be a parameter (lenght of nyu40
          // semantics)
          sum += (40.0 - static_cast<float>(tsdf_voxel->probabilities.size())) *
                 std::exp(tsdf_voxel->rest_probabilities - max_log_prob);
          max_prob = 1.0 / sum;
        }

        // we are pretty confident on the semantic estimation
        if (max_prob > 0.95) {
          // a bit looser geometric threshold here to avoid mistakenly remove
          // rich information
          if ((tsdf_voxel->geo_complexity < 0.8f * geo_thresholds[0])) {
            // process queried
            if (tsdf_voxel->small_type_status.is_this_type) {
              // it's no longer a small one
              tsdf_voxel->small_type_status.is_this_type = false;
              // its type updated
              tsdf_voxel->small_type_status.updated = true;
              if (!(tsdf_voxel->small_type_status.is_neighbor_of_this_type())) {
                // it's not a neighbor of small semantic, earse subvoxels
                std::vector<TsdfSubVoxel>().swap(
                    tsdf_voxel->child_voxels_queried);
              }
            }

            // process middle
            if (tsdf_voxel->middle_type_status.is_this_type) {
              // it's no longer a middle one
              tsdf_voxel->middle_type_status.is_this_type = false;
              // its type updated
              tsdf_voxel->middle_type_status.updated = true;
              if (!(tsdf_voxel->middle_type_status
                        .is_neighbor_of_this_type())) {
                // it's not a neighbor of middle semantic, earse subvoxels
                std::vector<TsdfSubVoxel>().swap(
                    tsdf_voxel->child_voxels_small);
              }
            }
          }
        }
      }
    }
  }

  tsdf_voxel->distance =
      (new_sdf > 0.0) ? std::min(config_.default_truncation_distance, new_sdf)
                      : std::max(-config_.default_truncation_distance, new_sdf);
  tsdf_voxel->weight = std::min(config_.max_weight, new_weight);
}

void TsdfIntegratorBase::fuseSemanticProbability(
    TsdfVoxel* tsdf_voxel, const uint32_t& semantic_top4_encoded,
    const uint32_t& probabilities_top4_encoded, const float& update_weight) {
  // Used to update probability for Bayesian method only
  float remained_probability = 1.0;
  const int NumberOfLabels = 40;
  int NumberOfLabelsRemained = NumberOfLabels;
  float probability_for_the_rest;

  voxblox::Semantics semantic_top4(4, 0);
  voxblox::SemanticProbabilities probabilities_top4(4, 0.0);

  semantic_top4[0] = semantic_top4_encoded & 0x000000FF;
  semantic_top4[1] = (semantic_top4_encoded & 0x0000FF00) >> 8;
  semantic_top4[2] = (semantic_top4_encoded & 0x00FF0000) >> 16;
  semantic_top4[3] = semantic_top4_encoded >> 24;

  if (semantic_top4[0] == 0) {
    // the point doesn't have valid semantic information
    return;
  }

  probabilities_top4[0] =
      (static_cast<float>(probabilities_top4_encoded & 0x000000FF)) / 255.0;
  probabilities_top4[1] =
      (static_cast<float>((probabilities_top4_encoded & 0x0000FF00) >> 8)) /
      255.0;
  probabilities_top4[2] =
      (static_cast<float>((probabilities_top4_encoded & 0x00FF0000) >> 16)) /
      255.0;
  probabilities_top4[3] =
      (static_cast<float>(probabilities_top4_encoded >> 24)) / 255.0;

  if (semantic_update_method == "bayesian") {
    for (int i = 0; i < 4; ++i) {
      if (semantic_top4[i] == 0) break;
      remained_probability -= probabilities_top4[i];
      NumberOfLabelsRemained -= 1;
    }
  }

  // TODO(jianhao): CHeck this, sometimes the value is even -0.1
  CHECK_GE(remained_probability, -0.2);
  // Assume the rest label averagely take the remaining
  // probability the probability will not be lower than 0.1%
  if (semantic_update_method == "bayesian") {
    probability_for_the_rest = std::max(
        remained_probability / ((float)NumberOfLabelsRemained), (float)0.01);
  } else if (semantic_update_method == "bayesian_constant") {
    probability_for_the_rest = 0.1;
  }

  for (int i = 0; i < 4; ++i) {
    uint32_t label = semantic_top4[i];
    if (label == 0) break;
    Semantics::iterator iter =
        std::find(tsdf_voxel->labels.begin(), tsdf_voxel->labels.end(), label);

    if (semantic_update_method == "max_pooling") {
      if (iter == tsdf_voxel->labels.end()) {
        tsdf_voxel->labels.push_back(label);
        tsdf_voxel->probabilities.push_back(0.1);
      } else {
        size_t idx = std::distance(tsdf_voxel->labels.begin(), iter);
        tsdf_voxel->probabilities[idx] += 0.1;
      }
    } else if (semantic_update_method == "weighted_max") {
      if (iter == tsdf_voxel->labels.end()) {
        tsdf_voxel->labels.push_back(label);
        tsdf_voxel->probabilities.push_back(probabilities_top4[i]);
      } else {
        size_t idx = std::distance(tsdf_voxel->labels.begin(), iter);
        tsdf_voxel->probabilities[idx] += probabilities_top4[i];
      }
    } else if (semantic_update_method == "bayesian" ||
               semantic_update_method == "bayesian_constant") {
      CHECK_GE(probabilities_top4[i], 0.09);

      /* log implementation */
      if (iter == tsdf_voxel->labels.end()) {
        tsdf_voxel->labels.push_back(label);
        tsdf_voxel->probabilities.push_back(
            tsdf_voxel->rest_probabilities +
            update_weight * (std::log(probabilities_top4[i]) -
                             std::log(probability_for_the_rest)));
      } else {
        size_t idx = std::distance(tsdf_voxel->labels.begin(), iter);
        tsdf_voxel->probabilities[idx] +=
            update_weight * (std::log(probabilities_top4[i]) -
                             std::log(probability_for_the_rest));
      }
    } else {
      std::cerr << "not implemented semantic_update_method: "
                << semantic_update_method << std::endl;
    }
  }

  tsdf_voxel->semantic_updated = true;
}

void TsdfIntegratorBase::fuseSemanticProbability(
    TsdfSubVoxel* tsdf_voxel, const uint32_t& semantic_top4_encoded,
    const uint32_t& probabilities_top4_encoded, const float& update_weight) {
  // Used to update probability for Bayesian method only
  float remained_probability = 1.0;
  const int NumberOfLabels = 40;
  int NumberOfLabelsRemained = NumberOfLabels;
  float probability_for_the_rest;

  voxblox::Semantics semantic_top4(4, 0);
  voxblox::SemanticProbabilities probabilities_top4(4, 0.0);

  semantic_top4[0] = semantic_top4_encoded & 0x000000FF;
  semantic_top4[1] = (semantic_top4_encoded & 0x0000FF00) >> 8;
  semantic_top4[2] = (semantic_top4_encoded & 0x00FF0000) >> 16;
  semantic_top4[3] = semantic_top4_encoded >> 24;

  if (semantic_top4[0] == 0) {
    // the point doesn't have valid semantic information
    return;
  }

  probabilities_top4[0] =
      (static_cast<float>(probabilities_top4_encoded & 0x000000FF)) / 255.0;
  probabilities_top4[1] =
      (static_cast<float>((probabilities_top4_encoded & 0x0000FF00) >> 8)) /
      255.0;
  probabilities_top4[2] =
      (static_cast<float>((probabilities_top4_encoded & 0x00FF0000) >> 16)) /
      255.0;
  probabilities_top4[3] =
      (static_cast<float>(probabilities_top4_encoded >> 24)) / 255.0;

  if (semantic_update_method == "bayesian") {
    for (int i = 0; i < 4; ++i) {
      if (semantic_top4[i] == 0) break;
      remained_probability -= probabilities_top4[i];
      NumberOfLabelsRemained -= 1;
    }
  }

  // TODO(jianhao): CHeck this, sometimes the value is even -0.1
  CHECK_GE(remained_probability, -0.2);
  // Assume the rest label averagely take the remaining
  // probability the probability will not be lower than 0.1%
  if (semantic_update_method == "bayesian") {
    probability_for_the_rest = std::max(
        remained_probability / ((float)NumberOfLabelsRemained), (float)0.01);
  } else if (semantic_update_method == "bayesian_constant") {
    probability_for_the_rest = 0.1;
  }

  for (int i = 0; i < 4; ++i) {
    uint32_t label = semantic_top4[i];
    if (label == 0) break;
    Semantics::iterator iter =
        std::find(tsdf_voxel->labels.begin(), tsdf_voxel->labels.end(), label);

    if (semantic_update_method == "max_pooling") {
      if (iter == tsdf_voxel->labels.end()) {
        tsdf_voxel->labels.push_back(label);
        tsdf_voxel->probabilities.push_back(0.1);
      } else {
        size_t idx = std::distance(tsdf_voxel->labels.begin(), iter);
        tsdf_voxel->probabilities[idx] += 0.1;
      }
    } else if (semantic_update_method == "weighted_max") {
      if (iter == tsdf_voxel->labels.end()) {
        tsdf_voxel->labels.push_back(label);
        tsdf_voxel->probabilities.push_back(probabilities_top4[i]);
      } else {
        size_t idx = std::distance(tsdf_voxel->labels.begin(), iter);
        tsdf_voxel->probabilities[idx] += probabilities_top4[i];
      }
    } else if (semantic_update_method == "bayesian" ||
               semantic_update_method == "bayesian_constant") {
      CHECK_GE(probabilities_top4[i], 0.09);

      /* log implementation */
      if (iter == tsdf_voxel->labels.end()) {
        tsdf_voxel->labels.push_back(label);
        tsdf_voxel->probabilities.push_back(
            tsdf_voxel->rest_probabilities +
            update_weight * (std::log(probabilities_top4[i]) -
                             std::log(probability_for_the_rest)));
      } else {
        size_t idx = std::distance(tsdf_voxel->labels.begin(), iter);
        tsdf_voxel->probabilities[idx] +=
            update_weight * (std::log(probabilities_top4[i]) -
                             std::log(probability_for_the_rest));
      }
    } else {
      std::cerr << "not implemented semantic_update_method: "
                << semantic_update_method << std::endl;
    }
  }
}

// Thread safe.
// Figure out whether the voxel is behind or in front of the surface.
// To do this, project the voxel_center onto the ray from origin to point
// G. Then check if the the magnitude of the vector is smaller or greater
// than the original distance...
float TsdfIntegratorBase::computeDistance(const Point& origin,
                                          const Point& point_G,
                                          const Point& voxel_center) const {
  const Point v_voxel_origin = voxel_center - origin;
  const Point v_point_origin = point_G - origin;

  const FloatingPoint dist_G = v_point_origin.norm();
  // projection of a (v_voxel_origin) onto b (v_point_origin)
  const FloatingPoint dist_G_V = v_voxel_origin.dot(v_point_origin) / dist_G;

  const float sdf = static_cast<float>(dist_G - dist_G_V);
  return sdf;
}

// Thread safe.
float TsdfIntegratorBase::getVoxelWeight(const Point& point_C) const {
  if (config_.use_const_weight) {
    return 1.0f;
  }
  const FloatingPoint dist_z = std::abs(point_C.z());
  if (dist_z > kEpsilon) {
    return 1.0f / (dist_z * dist_z);
  }
  return 0.0f;
}

void SimpleTsdfIntegrator::integratePointCloud(const Transformation& T_G_C,
                                               const Pointcloud& points_C,
                                               const Colors& colors,
                                               const bool freespace_points) {
  timing::Timer integrate_timer("integrate/simple");
  CHECK_EQ(points_C.size(), colors.size());

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  std::list<std::thread> integration_threads;
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(&SimpleTsdfIntegrator::integrateFunction,
                                     this, T_G_C, points_C, colors,
                                     freespace_points, index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }
  integrate_timer.Stop();

  timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();
  insertion_timer.Stop();
}

void SimpleTsdfIntegrator::integratePointCloudSemanticProbability(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool freespace_points) {
  std::cerr << "not implemented" << std::endl;
}

void SimpleTsdfIntegrator::integratePointcloudGeoComplexity(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const bool freespace_points) {
  std::cerr << "not implemented" << std::endl;
}

void SimpleTsdfIntegrator::integratePointcloudGeoSemantic(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool freespace_points) {
  std::cerr << "not implemented" << std::endl;
}

void SimpleTsdfIntegrator::integrateFunction(const Transformation& T_G_C,
                                             const Pointcloud& points_C,
                                             const Colors& colors,
                                             const bool freespace_points,
                                             ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx)) {
    const Point& point_C = points_C[point_idx];
    const Color& color = colors[point_idx];
    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point origin = T_G_C.getPosition();
    const Point point_G = T_G_C * point_C;

    RayCaster ray_caster(origin, point_G, is_clearing,
                         config_.voxel_carving_enabled,
                         config_.max_ray_length_m, voxel_size_inv_,
                         config_.default_truncation_distance);

    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    GlobalIndex global_voxel_idx;
    while (ray_caster.nextRayIndex(&global_voxel_idx)) {
      TsdfVoxel* voxel =
          allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

      const float weight = getVoxelWeight(point_C);

      updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, voxel);
    }
  }
}

void MergedTsdfIntegrator::integratePointCloud(const Transformation& T_G_C,
                                               const Pointcloud& points_C,
                                               const Colors& colors,
                                               const bool freespace_points) {
  timing::Timer integrate_timer("integrate/merged");
  CHECK_EQ(points_C.size(), colors.size());

  // Pre-compute a list of unique voxels to end on.
  // Create a hashmap: VOXEL INDEX -> index in original cloud.
  LongIndexHashMapType<AlignedVector<size_t>>::type voxel_map;
  // This is a hash map (same as above) to all the indices that
  // need to be cleared.
  LongIndexHashMapType<AlignedVector<size_t>>::type clear_map;

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  bundleRays(T_G_C, points_C, freespace_points, index_getter.get(), &voxel_map,
             &clear_map);

  integrateRays(T_G_C, points_C, colors, config_.enable_anti_grazing, false,
                voxel_map, clear_map);

  timing::Timer clear_timer("integrate/clear");

  integrateRays(T_G_C, points_C, colors, config_.enable_anti_grazing, true,
                voxel_map, clear_map);

  clear_timer.Stop();

  integrate_timer.Stop();
}

void MergedTsdfIntegrator::integratePointCloudSemanticProbability(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool freespace_points) {
  std::cerr << "not implemented" << std::endl;
}

void MergedTsdfIntegrator::integratePointcloudGeoComplexity(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const bool freespace_points) {
  std::cerr << "not implemented" << std::endl;
}

void MergedTsdfIntegrator::integratePointcloudGeoSemantic(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool freespace_points) {
  std::cerr << "not implemented" << std::endl;
}

void MergedTsdfIntegrator::bundleRays(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const bool freespace_points, ThreadSafeIndex* index_getter,
    LongIndexHashMapType<AlignedVector<size_t>>::type* voxel_map,
    LongIndexHashMapType<AlignedVector<size_t>>::type* clear_map) {
  DCHECK(voxel_map != nullptr);
  DCHECK(clear_map != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx)) {
    const Point& point_C = points_C[point_idx];
    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point point_G = T_G_C * point_C;

    GlobalIndex voxel_index =
        getGridIndexFromPoint<GlobalIndex>(point_G, voxel_size_inv_);

    if (is_clearing) {
      (*clear_map)[voxel_index].push_back(point_idx);
    } else {
      (*voxel_map)[voxel_index].push_back(point_idx);
    }
  }

  VLOG(3) << "Went from " << points_C.size() << " points to "
          << voxel_map->size() << " raycasts  and " << clear_map->size()
          << " clear rays.";
}

void MergedTsdfIntegrator::integrateVoxel(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, bool enable_anti_grazing, bool clearing_ray,
    const std::pair<GlobalIndex, AlignedVector<size_t>>& kv,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map) {
  if (kv.second.empty()) {
    return;
  }

  const Point& origin = T_G_C.getPosition();
  Color merged_color;
  Point merged_point_C = Point::Zero();
  FloatingPoint merged_weight = 0.0;

  for (const size_t pt_idx : kv.second) {
    const Point& point_C = points_C[pt_idx];
    const Color& color = colors[pt_idx];

    const float point_weight = getVoxelWeight(point_C);
    if (point_weight < kEpsilon) {
      continue;
    }
    merged_point_C = (merged_point_C * merged_weight + point_C * point_weight) /
                     (merged_weight + point_weight);
    merged_color =
        Color::blendTwoColors(merged_color, merged_weight, color, point_weight);
    merged_weight += point_weight;

    // only take first point when clearing
    if (clearing_ray) {
      break;
    }
  }

  const Point merged_point_G = T_G_C * merged_point_C;

  RayCaster ray_caster(origin, merged_point_G, clearing_ray,
                       config_.voxel_carving_enabled, config_.max_ray_length_m,
                       voxel_size_inv_, config_.default_truncation_distance);

  GlobalIndex global_voxel_idx;
  while (ray_caster.nextRayIndex(&global_voxel_idx)) {
    if (enable_anti_grazing) {
      // Check if this one is already the the block hash map for
      // this insertion. Skip this to avoid grazing.
      if ((clearing_ray || global_voxel_idx != kv.first) &&
          voxel_map.find(global_voxel_idx) != voxel_map.end()) {
        continue;
      }
    }

    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    TsdfVoxel* voxel =
        allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

    updateTsdfVoxel(origin, merged_point_G, global_voxel_idx, merged_color,
                    merged_weight, voxel);
  }
}

void MergedTsdfIntegrator::integrateVoxels(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, bool enable_anti_grazing, bool clearing_ray,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& clear_map,
    size_t thread_idx) {
  LongIndexHashMapType<AlignedVector<size_t>>::type::const_iterator it;
  size_t map_size;
  if (clearing_ray) {
    it = clear_map.begin();
    map_size = clear_map.size();
  } else {
    it = voxel_map.begin();
    map_size = voxel_map.size();
  }

  for (size_t i = 0; i < map_size; ++i) {
    if (((i + thread_idx + 1) % config_.integrator_threads) == 0) {
      integrateVoxel(T_G_C, points_C, colors, enable_anti_grazing, clearing_ray,
                     *it, voxel_map);
    }
    ++it;
  }
}

void MergedTsdfIntegrator::integrateRays(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, bool enable_anti_grazing, bool clearing_ray,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& clear_map) {
  // if only 1 thread just do function call, otherwise spawn
  // threads
  if (config_.integrator_threads == 1) {
    constexpr size_t thread_idx = 0;
    integrateVoxels(T_G_C, points_C, colors, enable_anti_grazing, clearing_ray,
                    voxel_map, clear_map, thread_idx);
  } else {
    std::list<std::thread> integration_threads;
    for (size_t i = 0; i < config_.integrator_threads; ++i) {
      integration_threads.emplace_back(
          &MergedTsdfIntegrator::integrateVoxels, this, T_G_C, points_C, colors,
          enable_anti_grazing, clearing_ray, voxel_map, clear_map, i);
    }

    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  }

  timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();

  insertion_timer.Stop();
}

void FastTsdfIntegrator::integrateFunction(const Transformation& T_G_C,
                                           const Pointcloud& points_C,
                                           const Colors& colors,
                                           const bool freespace_points,
                                           ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx) &&
         (std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - integration_start_time_)
              .count() < config_.max_integration_time_s * 1000000)) {
    const Point& point_C = points_C[point_idx];
    const Color& color = colors[point_idx];
    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point origin = T_G_C.getPosition();
    const Point point_G = T_G_C * point_C;
    // Checks to see if another ray in this scan has already
    // started 'close' to this location. If it has then we skip
    // ray casting this point. We measure if a start location is
    // 'close' to another points by inserting the point into a set
    // of voxels. This voxel set has a resolution
    // start_voxel_subsampling_factor times higher then the voxel
    // size.
    GlobalIndex global_voxel_idx;
    global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(
        point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_);
    if (!start_voxel_approx_set_.replaceHash(global_voxel_idx)) {
      continue;
    }

    constexpr bool cast_from_origin = false;
    RayCaster ray_caster(origin, point_G, is_clearing,
                         config_.voxel_carving_enabled,
                         config_.max_ray_length_m, voxel_size_inv_,
                         config_.default_truncation_distance, cast_from_origin);

    int64_t consecutive_ray_collisions = 0;

    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    while (ray_caster.nextRayIndex(&global_voxel_idx)) {
      // Check if the current voxel has been seen by any ray cast
      // this scan. If it has increment the
      // consecutive_ray_collisions counter, otherwise reset it.
      // If the counter reaches a threshold we stop casting as the
      // ray is deemed to be contributing too little new
      // information.
      if (!voxel_observed_approx_set_.replaceHash(global_voxel_idx)) {
        ++consecutive_ray_collisions;
      } else {
        consecutive_ray_collisions = 0;
      }
      if (consecutive_ray_collisions > config_.max_consecutive_ray_collisions) {
        break;
      }

      TsdfVoxel* voxel =
          allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

      const float weight = getVoxelWeight(point_C);

      updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, voxel);
    }
  }
}

void FastTsdfIntegrator::integrateFunctionSemanticProbability(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool freespace_points, ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx) &&
         (std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - integration_start_time_)
              .count() < config_.max_integration_time_s * 1000000)) {
    const Point& point_C = points_C[point_idx];
    const Color& color = colors[point_idx];

    const uint32_t& semantic_top4_encoded = semantics_encoded[point_idx];
    const uint32_t& probabilities_top4_encoded =
        probabilities_encoded[point_idx];

    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point origin = T_G_C.getPosition();
    const Point point_G = T_G_C * point_C;
    // Checks to see if another ray in this scan has already
    // started 'close' to this location. If it has then we skip
    // ray casting this point. We measure if a start location is
    // 'close' to another points by inserting the point into a set
    // of voxels. This voxel set has a resolution
    // start_voxel_subsampling_factor times higher then the voxel
    // size.
    GlobalIndex global_voxel_idx;
    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;

    constexpr bool cast_from_origin = false;

    global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(
        point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_);
    if (start_voxel_approx_set_.replaceHash(global_voxel_idx)) {
      RayCaster ray_caster(
          origin, point_G, is_clearing, config_.voxel_carving_enabled,
          config_.max_ray_length_m, voxel_size_inv_,
          config_.default_truncation_distance, cast_from_origin);

      int64_t consecutive_ray_collisions = 0;

      while (ray_caster.nextRayIndex(&global_voxel_idx)) {
        // Check if the current voxel has been seen by any ray cast
        // this scan. If it has increment the
        // consecutive_ray_collisions counter, otherwise reset it.
        // If the counter reaches a threshold we stop casting as the
        // ray is deemed to be contributing too little new
        // information.
        if (!voxel_observed_approx_set_.replaceHash(global_voxel_idx)) {
          ++consecutive_ray_collisions;
        } else {
          consecutive_ray_collisions = 0;
        }
        if (consecutive_ray_collisions >
            config_.max_consecutive_ray_collisions) {
          break;
        }

        TsdfVoxel* voxel =
            allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

        const float weight = getVoxelWeight(point_C);

        updateTsdfVoxelSemanticProbability(
            origin, point_G, global_voxel_idx, color, semantic_top4_encoded,
            probabilities_top4_encoded, weight, voxel);

        if (adaptive_mapping) {
          // check if  the type of this voxel has been recently update, we
          // should look its neighbors
          if (voxel->small_type_status.updated) {
            voxel->small_type_status.updated = false;

            Eigen::Matrix<LongIndexElement, 3, 26> cube_index_offsets;

            cube_index_offsets << -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0,
                1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1,
                1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1,
                0, 1, -1, 0, 1, -1, 0, 1;

            // if the voxel is one of the small semantics,
            // we also needs to subdivide its neighbors
            if (voxel->small_type_status.is_this_type) {
              for (unsigned int i = 0; i < 26; ++i) {
                GlobalIndex global_neighbor_voxel_idx =
                    global_voxel_idx + cube_index_offsets.col(i);

                std::lock_guard<std::mutex> lock(
                    mutexes_.get(global_neighbor_voxel_idx));

                TsdfVoxel* neighbor_voxel = allocateStorageAndGetVoxelPtr(
                    global_neighbor_voxel_idx, &block, &block_idx);

                // Check if this voxel has already been subdivided.
                // If so, we just need to update its status
                if (neighbor_voxel->small_type_status.should_be_divided()) {
                  // update the status of the neighbor voxel

                  // neighbor_voxel->small_type_status.neighbor_status |= (1 <<
                  // i);

                  neighbor_voxel->small_type_status.neighbor_status(i) = true;
                  continue;
                }

                // If not, we need to divide it and update its status
                // neighbor_voxel->small_type_status.neighbor_status |= (1 <<
                // i);

                neighbor_voxel->small_type_status.neighbor_status(i) = true;

                CHECK_EQ(neighbor_voxel->child_voxels_queried.size(), 0);
                std::vector<TsdfSubVoxel>().swap(
                    neighbor_voxel->child_voxels_queried);
                neighbor_voxel->child_voxels_queried.resize(
                    adaptive_ratio_small * adaptive_ratio_small *
                    adaptive_ratio_small);
              }
            } else {
              // If the voxel is found not toe be one of the small semantics,
              // we needs to update its neighbors
              for (unsigned int i = 0; i < 8; ++i) {
                GlobalIndex global_neighbor_voxel_idx =
                    global_voxel_idx + cube_index_offsets.col(i);

                std::lock_guard<std::mutex> lock(
                    mutexes_.get(global_neighbor_voxel_idx));

                TsdfVoxel* neighbor_voxel = allocateStorageAndGetVoxelPtr(
                    global_neighbor_voxel_idx, &block, &block_idx);

                // update the status of the neighbor voxel
                // neighbor_voxel->small_type_status.neighbor_status &= ~(1 <<
                // i);

                neighbor_voxel->small_type_status.neighbor_status(i) = false;

                if (!neighbor_voxel->small_type_status
                         .is_neighbor_of_this_type()) {
                  // it's no longer a neighbor of a small semantic
                  if (!(neighbor_voxel->small_type_status.is_this_type)) {
                    // we need to earse the subdivision
                    std::vector<TsdfSubVoxel>().swap(
                        neighbor_voxel->child_voxels_queried);
                  }
                }
              }
            }
          }
          if (voxel->middle_type_status.updated) {
            voxel->middle_type_status.updated = false;

            Eigen::Matrix<LongIndexElement, 3, 26> cube_index_offsets;
            cube_index_offsets << -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0,
                1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1,
                1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1,
                0, 1, -1, 0, 1, -1, 0, 1;

            // if the voxel is one of the middle semantics,
            // we also needs to subdivide its neighbors
            if (voxel->middle_type_status.is_this_type) {
              for (unsigned int i = 0; i < 26; ++i) {
                GlobalIndex global_neighbor_voxel_idx =
                    global_voxel_idx + cube_index_offsets.col(i);

                std::lock_guard<std::mutex> lock(
                    mutexes_.get(global_neighbor_voxel_idx));

                TsdfVoxel* neighbor_voxel = allocateStorageAndGetVoxelPtr(
                    global_neighbor_voxel_idx, &block, &block_idx);

                // Check if this voxel has already been subdivided.
                // If so, we just need to update its status
                if (neighbor_voxel->middle_type_status.should_be_divided()) {
                  // update the status of the neighbor voxel

                  // neighbor_voxel->middle_type_status.neighbor_status |= (1 <<
                  // i);

                  neighbor_voxel->middle_type_status.neighbor_status(i) = true;
                  continue;
                }

                // If not, we need to divide it and update its status

                // neighbor_voxel->middle_type_status.neighbor_status |= (1 <<
                // i);

                neighbor_voxel->middle_type_status.neighbor_status(i) = true;

                CHECK_EQ(neighbor_voxel->child_voxels_small.size(), 0);
                std::vector<TsdfSubVoxel>().swap(
                    neighbor_voxel->child_voxels_small);
                neighbor_voxel->child_voxels_small.resize(
                    adaptive_ratio_middle * adaptive_ratio_middle *
                    adaptive_ratio_middle);
              }
            } else {
              // If the voxel is found not toe be one of the small semantics,
              // we needs to update its neighbors
              for (unsigned int i = 0; i < 8; ++i) {
                GlobalIndex global_neighbor_voxel_idx =
                    global_voxel_idx + cube_index_offsets.col(i);

                std::lock_guard<std::mutex> lock(
                    mutexes_.get(global_neighbor_voxel_idx));

                TsdfVoxel* neighbor_voxel = allocateStorageAndGetVoxelPtr(
                    global_neighbor_voxel_idx, &block, &block_idx);

                // update the status of the neighbor voxel

                // neighbor_voxel->middle_type_status.neighbor_status &= ~(1 <<
                // i);
                neighbor_voxel->middle_type_status.neighbor_status(i) = false;

                if (!neighbor_voxel->middle_type_status
                         .is_neighbor_of_this_type()) {
                  // it's no longer a neighbor of a small semantic
                  if (!(neighbor_voxel->middle_type_status.is_this_type)) {
                    // we need to earse the subdivision
                    std::vector<TsdfSubVoxel>().swap(
                        neighbor_voxel->child_voxels_small);
                  }
                }
              }
            }
          }
        }
      }
    }

    if (adaptive_mapping) {
      // update small semantic voxels
      global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(
          point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_ *
                       static_cast<FloatingPoint>(adaptive_ratio_middle));
      if (start_voxel_approx_set_middle.replaceHash(global_voxel_idx)) {
        raycastingSubvoxels(origin, point_G, point_C, color,
                            semantic_top4_encoded, probabilities_top4_encoded,
                            is_clearing, cast_from_origin, false);
      }

      // update queried semantic voxels
      global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(
          point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_ *
                       static_cast<FloatingPoint>(adaptive_ratio_small));
      if (start_voxel_approx_set_small.replaceHash(global_voxel_idx)) {
        raycastingSubvoxels(origin, point_G, point_C, color,
                            semantic_top4_encoded, probabilities_top4_encoded,
                            is_clearing, cast_from_origin, true);
      }
    }
  }
}

void FastTsdfIntegrator::raycastingSubvoxels(
    const Point& origin, const Point& point_G, const Point& point_C,
    const Color& color, const uint32_t& semantic_top4_encoded,
    const uint32_t& probabilities_top4_encoded, const bool& is_clearing,
    const bool& cast_from_origin, const bool& update_queried) {
  DCHECK(!is_clearing);

  int voxel_size_ratio;
  float voxel_size_ratio_inv;
  if (update_queried) {
    // we are updating the subvoxel of queried semantic voxels
    voxel_size_ratio = adaptive_ratio_small;
  } else {
    // TODO(jianhao) since we only have two levels for now, if
    // it's not small semantic, then it's middle semantic
    voxel_size_ratio = adaptive_ratio_middle;
  }
  voxel_size_ratio_inv = 1.0f / static_cast<float>(voxel_size_ratio);

  // never enable voxel carving when update sub voxels
  const bool sub_voxel_carving_enabled = false;
  // used to update subvoxels
  RayCaster ray_caster_subvoxel(
      origin, point_G, is_clearing, sub_voxel_carving_enabled,
      config_.max_ray_length_m,
      static_cast<FloatingPoint>(voxel_size_ratio) * voxel_size_inv_,
      config_.default_truncation_distance /
          static_cast<FloatingPoint>(voxel_size_ratio),
      cast_from_origin);

  int64_t consecutive_ray_collisions = 0;

  GlobalIndex global_child_voxel_idx;
  TsdfVoxel* parent_voxel = nullptr;
  GlobalIndex global_parent_voxel_idx;

  Block<TsdfVoxel>::Ptr block = nullptr;
  BlockIndex block_idx;

  while (ray_caster_subvoxel.nextRayIndex(&global_child_voxel_idx)) {
    // recover parent voxel index
    const GlobalIndex this_global_parent_voxel_idx =
        getParentVoxelIndexFromGlobalVoxelIndex(global_child_voxel_idx,
                                                voxel_size_ratio_inv);

    // Check if the current voxel has been seen by any ray cast
    // this scan. If it has increment the
    // consecutive_ray_collisions counter, otherwise reset it. If
    // the counter reaches a threshold we stop casting as the ray
    // is deemed to be contributing too little new information.

    if (update_queried) {
      if (!small_sub_voxel_observed_approx_set_.replaceHash(
              global_child_voxel_idx)) {
        ++consecutive_ray_collisions;
      } else {
        consecutive_ray_collisions = 0;
      }
      if (consecutive_ray_collisions > config_.max_consecutive_ray_collisions) {
        break;
      }
    } else {
      if (!middle_sub_voxel_observed_approx_set_.replaceHash(
              global_child_voxel_idx)) {
        ++consecutive_ray_collisions;
      } else {
        consecutive_ray_collisions = 0;
      }
      if (consecutive_ray_collisions > config_.max_consecutive_ray_collisions) {
        break;
      }
    }

    std::lock_guard<std::mutex> lock(
        mutexes_.get(this_global_parent_voxel_idx));

    if (global_parent_voxel_idx != this_global_parent_voxel_idx ||
        parent_voxel == nullptr) {
      parent_voxel = allocateStorageAndGetVoxelPtr(this_global_parent_voxel_idx,
                                                   &block, &block_idx);
      global_parent_voxel_idx = this_global_parent_voxel_idx;
    }

    if (update_queried) {
      if (!(parent_voxel->small_type_status.should_be_divided())) {
        CHECK_EQ(parent_voxel->child_voxels_queried.size(), 0);
        continue;
      }
      CHECK_EQ(
          parent_voxel->child_voxels_queried.size(),
          adaptive_ratio_small * adaptive_ratio_small * adaptive_ratio_small);
    } else {
      if (!(parent_voxel->middle_type_status.should_be_divided())) {
        CHECK_EQ(parent_voxel->child_voxels_small.size(), 0);
        continue;
      }
      CHECK_EQ(parent_voxel->child_voxels_small.size(),
               adaptive_ratio_middle * adaptive_ratio_middle *
                   adaptive_ratio_middle);
    }

    int local_child_idx = getLocalChildVoxelIndexFromGlobalVoxelIndex(
        global_child_voxel_idx, voxel_size_ratio);
    TsdfSubVoxel* sub_voxel;
    if (update_queried) {
      sub_voxel = &(parent_voxel->child_voxels_queried[local_child_idx]);
    } else {
      sub_voxel = &(parent_voxel->child_voxels_small[local_child_idx]);
    }

    const float weight = getVoxelWeight(point_C);

    const Point sub_voxel_center = getCenterPointFromGridIndex(
        global_child_voxel_idx,
        voxel_size_ / static_cast<FloatingPoint>(voxel_size_ratio));
    updateTsdfSubVoxelSemanticProbability(
        origin, point_G, sub_voxel_center, color, semantic_top4_encoded,
        probabilities_top4_encoded, weight, sub_voxel, voxel_size_ratio);
  }
}

void FastTsdfIntegrator::integrateFunctionGeoComplexity(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const bool freespace_points, ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx) &&
         (std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - integration_start_time_)
              .count() < config_.max_integration_time_s * 1000000)) {
    const Point& point_C = points_C[point_idx];
    const Color& color = colors[point_idx];
    const float& this_geo_complexity = geo_complexity[point_idx];
    if (this_geo_complexity > 0.33f) {
      std::cerr << "wrong this_geo_complexity: " << this_geo_complexity
                << std::endl;
    }
    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point origin = T_G_C.getPosition();
    const Point point_G = T_G_C * point_C;
    // Checks to see if another ray in this scan has already
    // started 'close' to this location. If it has then we skip
    // ray casting this point. We measure if a start location is
    // 'close' to another points by inserting the point into a set
    // of voxels. This voxel set has a resolution
    // start_voxel_subsampling_factor times higher then the voxel
    // size.
    GlobalIndex global_voxel_idx;
    global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(
        point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_);
    if (!start_voxel_approx_set_.replaceHash(global_voxel_idx)) {
      continue;
    }

    constexpr bool cast_from_origin = false;
    RayCaster ray_caster(origin, point_G, is_clearing,
                         config_.voxel_carving_enabled,
                         config_.max_ray_length_m, voxel_size_inv_,
                         config_.default_truncation_distance, cast_from_origin);

    int64_t consecutive_ray_collisions = 0;

    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    while (ray_caster.nextRayIndex(&global_voxel_idx)) {
      // Check if the current voxel has been seen by any ray cast
      // this scan. If it has increment the
      // consecutive_ray_collisions counter, otherwise reset it.
      // If the counter reaches a threshold we stop casting as the
      // ray is deemed to be contributing too little new
      // information.
      if (!voxel_observed_approx_set_.replaceHash(global_voxel_idx)) {
        ++consecutive_ray_collisions;
      } else {
        consecutive_ray_collisions = 0;
      }
      if (consecutive_ray_collisions > config_.max_consecutive_ray_collisions) {
        break;
      }

      TsdfVoxel* voxel =
          allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

      const float weight = getVoxelWeight(point_C);

      updateTsdfVoxelGeoComplexity(origin, point_G, global_voxel_idx, color,
                                   this_geo_complexity, weight, voxel);
    }
  }
}

void FastTsdfIntegrator::integrateFunctionGeoSemantic(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool freespace_points, ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx) &&
         (std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - integration_start_time_)
              .count() < config_.max_integration_time_s * 1000000)) {
    const Point& point_C = points_C[point_idx];
    const Color& color = colors[point_idx];

    const float& this_geo_complexity = geo_complexity[point_idx];
    if (this_geo_complexity > 0.33333333f) {
      std::cerr << "wrong this_geo_complexity: " << this_geo_complexity
                << std::endl;
    }

    const uint32_t& semantic_top4_encoded = semantics_encoded[point_idx];
    const uint32_t& probabilities_top4_encoded =
        probabilities_encoded[point_idx];

    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point origin = T_G_C.getPosition();
    const Point point_G = T_G_C * point_C;
    // Checks to see if another ray in this scan has already
    // started 'close' to this location. If it has then we skip
    // ray casting this point. We measure if a start location is
    // 'close' to another points by inserting the point into a set
    // of voxels. This voxel set has a resolution
    // start_voxel_subsampling_factor times higher then the voxel
    // size.
    GlobalIndex global_voxel_idx;
    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;

    constexpr bool cast_from_origin = false;

    global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(
        point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_);

    if (start_voxel_approx_set_.replaceHash(global_voxel_idx)) {
      RayCaster ray_caster(
          origin, point_G, is_clearing, config_.voxel_carving_enabled,
          config_.max_ray_length_m, voxel_size_inv_,
          config_.default_truncation_distance, cast_from_origin);

      int64_t consecutive_ray_collisions = 0;

      while (ray_caster.nextRayIndex(&global_voxel_idx)) {
        // Check if the current voxel has been seen by any ray cast
        // this scan. If it has increment the
        // consecutive_ray_collisions counter, otherwise reset it.
        // If the counter reaches a threshold we stop casting as the
        // ray is deemed to be contributing too little new
        // information.
        if (!voxel_observed_approx_set_.replaceHash(global_voxel_idx)) {
          ++consecutive_ray_collisions;
        } else {
          consecutive_ray_collisions = 0;
        }
        if (consecutive_ray_collisions >
            config_.max_consecutive_ray_collisions) {
          break;
        }

        TsdfVoxel* voxel =
            allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

        const float weight = getVoxelWeight(point_C);

        updateTsdfVoxelGeoSemantic(origin, point_G, global_voxel_idx, color,
                                   this_geo_complexity, semantic_top4_encoded,
                                   probabilities_top4_encoded, weight, voxel);

        if (adaptive_mapping) {
          // check if  the type of this voxel has been recently update, we
          // should look its neighbors
          if (voxel->small_type_status.updated) {
            voxel->small_type_status.updated = false;

            Eigen::Matrix<LongIndexElement, 3, 26> cube_index_offsets;
            cube_index_offsets << -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0,
                1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1,
                1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1,
                0, 1, -1, 0, 1, -1, 0, 1;

            // if the voxel is one of the small semantics,
            // we also needs to subdivide its neighbors
            if (voxel->small_type_status.is_this_type) {
              for (unsigned int i = 0; i < 26; ++i) {
                GlobalIndex global_neighbor_voxel_idx =
                    global_voxel_idx + cube_index_offsets.col(i);

                std::lock_guard<std::mutex> lock(
                    mutexes_.get(global_neighbor_voxel_idx));

                TsdfVoxel* neighbor_voxel = allocateStorageAndGetVoxelPtr(
                    global_neighbor_voxel_idx, &block, &block_idx);

                // Check if this voxel has already been subdivided.
                // If so, we just need to update its status
                if (neighbor_voxel->small_type_status.should_be_divided()) {
                  // update the status of the neighbor voxel

                  // neighbor_voxel->small_type_status.neighbor_status |= (1 <<
                  // i);

                  neighbor_voxel->small_type_status.neighbor_status(i) = true;
                  continue;
                }

                // If not, we need to divide it and update its status

                // neighbor_voxel->small_type_status.neighbor_status |= (1 <<
                // i);

                neighbor_voxel->small_type_status.neighbor_status(i) = true;

                CHECK_EQ(neighbor_voxel->child_voxels_queried.size(), 0);
                std::vector<TsdfSubVoxel>().swap(
                    neighbor_voxel->child_voxels_queried);
                neighbor_voxel->child_voxels_queried.resize(
                    adaptive_ratio_small * adaptive_ratio_small *
                    adaptive_ratio_small);
              }
            } else {
              // If the voxel is found not toe be one of the small semantics,
              // we needs to update its neighbors
              for (unsigned int i = 0; i < 8; ++i) {
                GlobalIndex global_neighbor_voxel_idx =
                    global_voxel_idx + cube_index_offsets.col(i);

                std::lock_guard<std::mutex> lock(
                    mutexes_.get(global_neighbor_voxel_idx));

                TsdfVoxel* neighbor_voxel = allocateStorageAndGetVoxelPtr(
                    global_neighbor_voxel_idx, &block, &block_idx);

                // update the status of the neighbor voxel
                // neighbor_voxel->small_type_status.neighbor_status &= ~(1 <<
                // i);

                neighbor_voxel->small_type_status.neighbor_status(i) = false;

                if (!neighbor_voxel->small_type_status
                         .is_neighbor_of_this_type()) {
                  // it's no longer a neighbor of a small semantic
                  if (!(neighbor_voxel->small_type_status.is_this_type)) {
                    // we need to earse the subdivision
                    std::vector<TsdfSubVoxel>().swap(
                        neighbor_voxel->child_voxels_queried);
                  }
                }
              }
            }
          }
          if (voxel->middle_type_status.updated) {
            voxel->middle_type_status.updated = false;

            Eigen::Matrix<LongIndexElement, 3, 26> cube_index_offsets;
            cube_index_offsets << -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0,
                1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1,
                1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1,
                0, 1, -1, 0, 1, -1, 0, 1;

            // if the voxel is one of the middle semantics,
            // we also needs to subdivide its neighbors
            if (voxel->middle_type_status.is_this_type) {
              for (unsigned int i = 0; i < 26; ++i) {
                GlobalIndex global_neighbor_voxel_idx =
                    global_voxel_idx + cube_index_offsets.col(i);

                std::lock_guard<std::mutex> lock(
                    mutexes_.get(global_neighbor_voxel_idx));

                TsdfVoxel* neighbor_voxel = allocateStorageAndGetVoxelPtr(
                    global_neighbor_voxel_idx, &block, &block_idx);

                // Check if this voxel has already been subdivided.
                // If so, we just need to update its status
                if (neighbor_voxel->middle_type_status.should_be_divided()) {
                  // update the status of the neighbor voxel

                  // neighbor_voxel->middle_type_status.neighbor_status |= (1 <<
                  // i);

                  neighbor_voxel->middle_type_status.neighbor_status(i) = true;
                  continue;
                }

                // If not, we need to divide it and update its status

                // neighbor_voxel->middle_type_status.neighbor_status |= (1 <<
                // i);

                neighbor_voxel->middle_type_status.neighbor_status(i) = true;

                CHECK_EQ(neighbor_voxel->child_voxels_small.size(), 0);
                std::vector<TsdfSubVoxel>().swap(
                    neighbor_voxel->child_voxels_small);
                neighbor_voxel->child_voxels_small.resize(
                    adaptive_ratio_middle * adaptive_ratio_middle *
                    adaptive_ratio_middle);
              }
            } else {
              // If the voxel is found not toe be one of the small semantics,
              // we needs to update its neighbors
              for (unsigned int i = 0; i < 8; ++i) {
                GlobalIndex global_neighbor_voxel_idx =
                    global_voxel_idx + cube_index_offsets.col(i);

                std::lock_guard<std::mutex> lock(
                    mutexes_.get(global_neighbor_voxel_idx));

                TsdfVoxel* neighbor_voxel = allocateStorageAndGetVoxelPtr(
                    global_neighbor_voxel_idx, &block, &block_idx);

                // update the status of the neighbor voxel

                // neighbor_voxel->middle_type_status.neighbor_status &= ~(1 <<
                // i);
                neighbor_voxel->middle_type_status.neighbor_status(i) = false;

                if (!neighbor_voxel->middle_type_status
                         .is_neighbor_of_this_type()) {
                  // it's no longer a neighbor of a small semantic
                  if (!(neighbor_voxel->middle_type_status.is_this_type)) {
                    // we need to earse the subdivision
                    std::vector<TsdfSubVoxel>().swap(
                        neighbor_voxel->child_voxels_small);
                  }
                }
              }
            }
          }
        }
      }
    }

    if (adaptive_mapping) {
      // update small semantic voxels
      global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(
          point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_ *
                       static_cast<FloatingPoint>(adaptive_ratio_middle));
      if (start_voxel_approx_set_middle.replaceHash(global_voxel_idx)) {
        raycastingSubvoxels(origin, point_G, point_C, color,
                            semantic_top4_encoded, probabilities_top4_encoded,
                            is_clearing, cast_from_origin, false);
      }

      // update queried semantic voxels
      global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(
          point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_ *
                       static_cast<FloatingPoint>(adaptive_ratio_small));
      if (start_voxel_approx_set_small.replaceHash(global_voxel_idx)) {
        raycastingSubvoxels(origin, point_G, point_C, color,
                            semantic_top4_encoded, probabilities_top4_encoded,
                            is_clearing, cast_from_origin, true);
      }
    }
  }
}

void FastTsdfIntegrator::integratePointCloud(const Transformation& T_G_C,
                                             const Pointcloud& points_C,
                                             const Colors& colors,
                                             const bool freespace_points) {
  timing::Timer integrate_timer("integrate/fast");
  CHECK_EQ(points_C.size(), colors.size());

  integration_start_time_ = std::chrono::steady_clock::now();

  static int64_t reset_counter = 0;
  if ((++reset_counter) >= config_.clear_checks_every_n_frames) {
    reset_counter = 0;
    start_voxel_approx_set_.resetApproxSet();
    voxel_observed_approx_set_.resetApproxSet();
  }

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  std::list<std::thread> integration_threads;
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(&FastTsdfIntegrator::integrateFunction,
                                     this, T_G_C, points_C, colors,
                                     freespace_points, index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }

  integrate_timer.Stop();

  timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();
  insertion_timer.Stop();
}

void FastTsdfIntegrator::integratePointCloudSemanticProbability(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool freespace_points) {
  timing::Timer integrate_timer("integrate/fast");
  CHECK(adaptive_ratio_small % adaptive_ratio_middle == 0);

  integration_start_time_ = std::chrono::steady_clock::now();

  static int64_t reset_counter = 0;
  if ((++reset_counter) >= config_.clear_checks_every_n_frames) {
    reset_counter = 0;
    start_voxel_approx_set_.resetApproxSet();
    start_voxel_approx_set_middle.resetApproxSet();
    start_voxel_approx_set_small.resetApproxSet();
    voxel_observed_approx_set_.resetApproxSet();
    small_sub_voxel_observed_approx_set_.resetApproxSet();
    middle_sub_voxel_observed_approx_set_.resetApproxSet();
  }

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  std::list<std::thread> integration_threads;
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(
        &FastTsdfIntegrator::integrateFunctionSemanticProbability, this, T_G_C,
        points_C, colors, semantics_encoded, probabilities_encoded,
        freespace_points, index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }

  integrate_timer.Stop();

  timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();
  insertion_timer.Stop();
}

void FastTsdfIntegrator::integratePointcloudGeoComplexity(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const bool freespace_points) {
  timing::Timer integrate_timer("integrate/fast");

  std::cerr << "This function hasn't been developed" << std::endl;

  integration_start_time_ = std::chrono::steady_clock::now();

  static int64_t reset_counter = 0;
  if ((++reset_counter) >= config_.clear_checks_every_n_frames) {
    reset_counter = 0;
    start_voxel_approx_set_.resetApproxSet();
    start_voxel_approx_set_middle.resetApproxSet();
    start_voxel_approx_set_small.resetApproxSet();
    voxel_observed_approx_set_.resetApproxSet();
    small_sub_voxel_observed_approx_set_.resetApproxSet();
    middle_sub_voxel_observed_approx_set_.resetApproxSet();
  }

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  std::list<std::thread> integration_threads;
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(
        &FastTsdfIntegrator::integrateFunctionGeoComplexity, this, T_G_C,
        points_C, colors, geo_complexity, freespace_points, index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }

  integrate_timer.Stop();

  timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();
  insertion_timer.Stop();
}

void FastTsdfIntegrator::integratePointcloudGeoSemantic(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool freespace_points) {
  timing::Timer integrate_timer("integrate/fast");

  integration_start_time_ = std::chrono::steady_clock::now();

  static int64_t reset_counter = 0;
  if ((++reset_counter) >= config_.clear_checks_every_n_frames) {
    reset_counter = 0;
    start_voxel_approx_set_.resetApproxSet();
    start_voxel_approx_set_middle.resetApproxSet();
    start_voxel_approx_set_small.resetApproxSet();
    voxel_observed_approx_set_.resetApproxSet();
    small_sub_voxel_observed_approx_set_.resetApproxSet();
    middle_sub_voxel_observed_approx_set_.resetApproxSet();
  }

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  std::list<std::thread> integration_threads;
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(
        &FastTsdfIntegrator::integrateFunctionGeoSemantic, this, T_G_C,
        points_C, colors, geo_complexity, semantics_encoded,
        probabilities_encoded, freespace_points, index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }

  integrate_timer.Stop();

  timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();
  insertion_timer.Stop();
}

std::string TsdfIntegratorBase::Config::print() const {
  std::stringstream ss;
  // clang-format off
  ss << "================== TSDF Integrator Config ====================\n";
  ss << " General: \n";
  ss << " - default_truncation_distance:               " << default_truncation_distance << "\n";
  ss << " - max_weight:                                " << max_weight << "\n";
  ss << " - voxel_carving_enabled:                     " << voxel_carving_enabled << "\n";
  ss << " - min_ray_length_m:                          " << min_ray_length_m << "\n";
  ss << " - max_ray_length_m:                          " << max_ray_length_m << "\n";
  ss << " - use_const_weight:                          " << use_const_weight << "\n";
  ss << " - allow_clear:                               " << allow_clear << "\n";
  ss << " - use_weight_dropoff:                        " << use_weight_dropoff << "\n";
  ss << " - use_sparsity_compensation_factor:          " << use_sparsity_compensation_factor << "\n";
  ss << " - sparsity_compensation_factor:              "  << sparsity_compensation_factor << "\n";
  ss << " - integrator_threads:                        " << integrator_threads << "\n";
  ss << " MergedTsdfIntegrator: \n";
  ss << " - enable_anti_grazing:                       " << enable_anti_grazing << "\n";
  ss << " FastTsdfIntegrator: \n";
  ss << " - start_voxel_subsampling_factor:            " << start_voxel_subsampling_factor << "\n";
  ss << " - max_consecutive_ray_collisions:            " << max_consecutive_ray_collisions << "\n";
  ss << " - clear_checks_every_n_frames:               " << clear_checks_every_n_frames << "\n";
  ss << " - max_integration_time_s:                    " << max_integration_time_s << "\n";
  ss << "==============================================================\n";
  // clang-format on
  return ss.str();
}

}  // namespace voxblox
