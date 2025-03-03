// The MIT License (MIT)
// Copyright (c) 2014 Matthew Klingensmith and Ivan Dryanovski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef VOXBLOX_MESH_MESH_INTEGRATOR_H_
#define VOXBLOX_MESH_MESH_INTEGRATOR_H_

#include <glog/logging.h>
#include <voxblox/utils/color_maps.h>

#include <Eigen/Core>
#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "voxblox/core/common.h"
#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"
#include "voxblox/integrator/integrator_utils.h"
#include "voxblox/interpolator/interpolator.h"
#include "voxblox/mesh/marching_cubes.h"
#include "voxblox/mesh/mesh_layer.h"
#include "voxblox/utils/meshing_utils.h"
#include "voxblox/utils/timing.h"

namespace voxblox {

struct MeshIntegratorConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool use_color = true;
  float min_weight = 1e-4;

  size_t integrator_threads = std::thread::hardware_concurrency();

  inline std::string print() const {
    std::stringstream ss;
    // clang-format off
    ss << "================== Mesh Integrator Config ====================\n";
    ss << " - use_color:                 " << use_color << "\n";
    ss << " - min_weight:                " << min_weight << "\n";
    ss << " - integrator_threads:        " << integrator_threads << "\n";
    ss << "==============================================================\n";
    // clang-format on
    return ss.str();
  }
};

/**
 * Integrates a TSDF layer to incrementally update a mesh layer using marching
 * cubes.
 */
template <typename VoxelType>
class MeshIntegrator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool adaptive_mapping = false;
  int adaptive_ratio_small;   // must be an integer
  int adaptive_ratio_middle;  // must be an integer
  std::vector<uint32_t> small_semantic;
  std::vector<uint32_t> large_semantic;

  void initFromSdfLayer(const Layer<VoxelType>& sdf_layer) {
    voxel_size_ = sdf_layer.voxel_size();
    block_size_ = sdf_layer.block_size();
    voxels_per_side_ = sdf_layer.voxels_per_side();

    voxel_size_inv_ = 1.0 / voxel_size_;
    block_size_inv_ = 1.0 / block_size_;
    voxels_per_side_inv_ = 1.0 / voxels_per_side_;
  }

  /**
   * Use this constructor in case you would like to modify the layer during mesh
   * extraction, i.e. modify the updated flag.
   */
  MeshIntegrator(const MeshIntegratorConfig& config,
                 Layer<VoxelType>* sdf_layer, MeshLayer* mesh_layer)
      : config_(config),
        sdf_layer_mutable_(CHECK_NOTNULL(sdf_layer)),
        sdf_layer_const_(CHECK_NOTNULL(sdf_layer)),
        mesh_layer_(CHECK_NOTNULL(mesh_layer)) {
    initFromSdfLayer(*sdf_layer);

    cube_index_offsets_ << 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1;

    cube_index_face_ << 0, 3, 4, 7, 0, 1, 4, 5, 0, 1, 2, 3, 1, 2, 5, 6, 3, 2, 7,
        6, 4, 5, 6, 7;
    axis_index_face_ << 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1;
    cube_index_edge_ << 0, 1, 0, 3, 0, 4, 4, 5, 4, 7, 3, 7, 3, 2, 1, 2, 1, 5, 7,
        6, 5, 6, 2, 6;
    axis_index_edge_ << 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2;

    if (config_.integrator_threads == 0) {
      LOG(WARNING) << "Automatic core count failed, defaulting to 1 threads";
      config_.integrator_threads = 1;
    }
  }

  /**
   * This constructor will not allow you to modify the layer, i.e. clear the
   * updated flag.
   */
  MeshIntegrator(const MeshIntegratorConfig& config,
                 const Layer<VoxelType>& sdf_layer, MeshLayer* mesh_layer)
      : config_(config),
        sdf_layer_mutable_(nullptr),
        sdf_layer_const_(&sdf_layer),
        mesh_layer_(CHECK_NOTNULL(mesh_layer)) {
    initFromSdfLayer(sdf_layer);

    // clang-format off
    cube_index_offsets_ << 0, 1, 1, 0, 0, 1, 1, 0,
                           0, 0, 1, 1, 0, 0, 1, 1,
                           0, 0, 0, 0, 1, 1, 1, 1;
    // clang-format on

    // clang-format off
    cube_index_face_<< 0, 3, 4, 7, 0, 1, 4, 5, 0, 1, 2, 3,
                       1, 2, 5, 6, 3, 2, 7, 6, 4, 5, 6, 7;
    axis_index_face_<< 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                       1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1;
    cube_index_edge_<< 0, 1, 0, 3, 0, 4,
                       4, 5, 4, 7, 3, 7,
                       3, 2, 1, 2, 1, 5,
                       7, 6, 5, 6, 2, 6;
    axis_index_edge_<< 1, 1, 0, 0, 0, 0,
                       2, 2, 2, 2, 1, 1,
                       0, 0, 1, 1, 2, 2;
    // clang-format on

    if (config_.integrator_threads == 0) {
      LOG(WARNING) << "Automatic core count failed, defaulting to 1 threads";
      config_.integrator_threads = 1;
    }
  }

  /// Generates mesh from the tsdf layer.
  void generateMesh(bool only_mesh_updated_blocks, bool clear_updated_flag) {
    CHECK(!clear_updated_flag || (sdf_layer_mutable_ != nullptr))
        << "If you would like to modify the updated flag in the blocks, please "
        << "use the constructor that provides a non-const link to the sdf "
        << "layer!";
    BlockIndexList all_tsdf_blocks;
    if (only_mesh_updated_blocks) {
      sdf_layer_const_->getAllUpdatedBlocks(Update::kMesh, &all_tsdf_blocks);
    } else {
      sdf_layer_const_->getAllAllocatedBlocks(&all_tsdf_blocks);
    }

    // Allocate all the mesh memory
    for (const BlockIndex& block_index : all_tsdf_blocks) {
      mesh_layer_->allocateMeshPtrByIndex(block_index);
    }

    std::unique_ptr<ThreadSafeIndex> index_getter(
        new MixedThreadSafeIndex(all_tsdf_blocks.size()));

    std::list<std::thread> integration_threads;
    for (size_t i = 0; i < config_.integrator_threads; ++i) {
      integration_threads.emplace_back(
          &MeshIntegrator::generateMeshBlocksFunction, this, all_tsdf_blocks,
          clear_updated_flag, index_getter.get());
    }

    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  }
  /// Used for debugging
  void generateMeshSelective(bool only_mesh_updated_blocks,
                             bool clear_updated_flag, const int& type_id) {
    CHECK(!clear_updated_flag || (sdf_layer_mutable_ != nullptr))
        << "If you would like to modify the updated flag in the blocks, please "
        << "use the constructor that provides a non-const link to the sdf "
        << "layer!";
    BlockIndexList all_tsdf_blocks;
    if (only_mesh_updated_blocks) {
      sdf_layer_const_->getAllUpdatedBlocks(Update::kMesh, &all_tsdf_blocks);
    } else {
      sdf_layer_const_->getAllAllocatedBlocks(&all_tsdf_blocks);
    }

    // Allocate all the mesh memory
    for (const BlockIndex& block_index : all_tsdf_blocks) {
      mesh_layer_->allocateMeshPtrByIndex(block_index);
    }

    std::unique_ptr<ThreadSafeIndex> index_getter(
        new MixedThreadSafeIndex(all_tsdf_blocks.size()));

    std::list<std::thread> integration_threads;
    for (size_t i = 0; i < config_.integrator_threads; ++i) {
      integration_threads.emplace_back(
          &MeshIntegrator::generateMeshBlocksFunctionSelective, this,
          all_tsdf_blocks, clear_updated_flag, type_id, index_getter.get());
    }

    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  }

  /// Generates mesh coloered with max pooling label.
  void generateMeshSemantic(
      bool only_mesh_updated_blocks, bool clear_updated_flag,
      const std::shared_ptr<voxblox::ColorMap>& color_map) {
    CHECK(!clear_updated_flag || (sdf_layer_mutable_ != nullptr))
        << "If you would like to modify the updated flag in the blocks, please "
        << "use the constructor that provides a non-const link to the sdf "
        << "layer!";
    BlockIndexList all_tsdf_blocks;
    if (only_mesh_updated_blocks) {
      sdf_layer_const_->getAllUpdatedBlocks(Update::kMesh, &all_tsdf_blocks);
    } else {
      sdf_layer_const_->getAllAllocatedBlocks(&all_tsdf_blocks);
    }

    // Allocate all the mesh memory
    for (const BlockIndex& block_index : all_tsdf_blocks) {
      mesh_layer_->allocateMeshPtrByIndex(block_index);
    }

    std::unique_ptr<ThreadSafeIndex> index_getter(
        new MixedThreadSafeIndex(all_tsdf_blocks.size()));

    std::list<std::thread> integration_threads;
    for (size_t i = 0; i < config_.integrator_threads; ++i) {
      integration_threads.emplace_back(
          &MeshIntegrator::generateMeshBlocksFunctionSemantic, this,
          all_tsdf_blocks, clear_updated_flag, color_map, index_getter.get());
    }

    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  }

  void generateMeshBlocksFunction(const BlockIndexList& all_tsdf_blocks,
                                  bool clear_updated_flag,
                                  ThreadSafeIndex* index_getter) {
    DCHECK(index_getter != nullptr);
    CHECK(!clear_updated_flag || (sdf_layer_mutable_ != nullptr))
        << "If you would like to modify the updated flag in the blocks, please "
        << "use the constructor that provides a non-const link to the sdf "
        << "layer!";

    size_t list_idx;
    while (index_getter->getNextIndex(&list_idx)) {
      const BlockIndex& block_idx = all_tsdf_blocks[list_idx];
      updateMeshForBlock(block_idx);
      if (clear_updated_flag) {
        typename Block<VoxelType>::Ptr block =
            sdf_layer_mutable_->getBlockPtrByIndex(block_idx);
        block->updated().reset(Update::kMesh);
      }
    }
  }

  void generateMeshBlocksFunctionSelective(
      const BlockIndexList& all_tsdf_blocks, bool clear_updated_flag,
      const int& type_id, ThreadSafeIndex* index_getter) {
    DCHECK(index_getter != nullptr);
    CHECK(!clear_updated_flag || (sdf_layer_mutable_ != nullptr))
        << "If you would like to modify the updated flag in the blocks, please "
        << "use the constructor that provides a non-const link to the sdf "
        << "layer!";

    size_t list_idx;
    while (index_getter->getNextIndex(&list_idx)) {
      const BlockIndex& block_idx = all_tsdf_blocks[list_idx];
      updateMeshForBlockSelective(block_idx, type_id);
      if (clear_updated_flag) {
        typename Block<VoxelType>::Ptr block =
            sdf_layer_mutable_->getBlockPtrByIndex(block_idx);
        block->updated().reset(Update::kMesh);
      }
    }
  }

  void generateMeshBlocksFunctionSemantic(
      const BlockIndexList& all_tsdf_blocks, bool clear_updated_flag,
      const std::shared_ptr<voxblox::ColorMap>& color_map,
      ThreadSafeIndex* index_getter) {
    DCHECK(index_getter != nullptr);
    CHECK(!clear_updated_flag || (sdf_layer_mutable_ != nullptr))
        << "If you would like to modify the updated flag in the blocks, please "
        << "use the constructor that provides a non-const link to the sdf "
        << "layer!";

    size_t list_idx;
    while (index_getter->getNextIndex(&list_idx)) {
      const BlockIndex& block_idx = all_tsdf_blocks[list_idx];
      updateMeshForBlockSemantic(block_idx, color_map);
      if (clear_updated_flag) {
        typename Block<VoxelType>::Ptr block =
            sdf_layer_mutable_->getBlockPtrByIndex(block_idx);
        block->updated().reset(Update::kMesh);
      }
    }
  }

  void extractBlockMesh(typename Block<VoxelType>::ConstPtr block,
                        Mesh::Ptr mesh) {
    DCHECK(block != nullptr);
    DCHECK(mesh != nullptr);

    IndexElement vps = block->voxels_per_side();
    VertexIndex next_mesh_index = 0;

    VoxelIndex voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < vps - 1; ++voxel_index.x()) {
      for (voxel_index.y() = 0; voxel_index.y() < vps - 1; ++voxel_index.y()) {
        for (voxel_index.z() = 0; voxel_index.z() < vps - 1;
             ++voxel_index.z()) {
          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
          extractMeshInsideBlock(*block, voxel_index, coords, &next_mesh_index,
                                 mesh.get());
        }
      }
    }

    // Max X plane
    // takes care of edge (x_max, y_max, z),
    // takes care of edge (x_max, y, z_max).
    voxel_index.x() = vps - 1;
    for (voxel_index.z() = 0; voxel_index.z() < vps; voxel_index.z()++) {
      for (voxel_index.y() = 0; voxel_index.y() < vps; voxel_index.y()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                            mesh.get());
      }
    }

    // Max Y plane.
    // takes care of edge (x, y_max, z_max),
    // without corner (x_max, y_max, z_max).
    voxel_index.y() = vps - 1;
    for (voxel_index.z() = 0; voxel_index.z() < vps; voxel_index.z()++) {
      for (voxel_index.x() = 0; voxel_index.x() < vps - 1; voxel_index.x()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                            mesh.get());
      }
    }

    // Max Z plane.
    voxel_index.z() = vps - 1;
    for (voxel_index.y() = 0; voxel_index.y() < vps - 1; voxel_index.y()++) {
      for (voxel_index.x() = 0; voxel_index.x() < vps - 1; voxel_index.x()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                            mesh.get());
      }
    }

    if (adaptive_mapping) {
      for (voxel_index.x() = 0; voxel_index.x() < vps; ++voxel_index.x()) {
        for (voxel_index.y() = 0; voxel_index.y() < vps; ++voxel_index.y()) {
          for (voxel_index.z() = 0; voxel_index.z() < vps; ++voxel_index.z()) {
            Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
            const VoxelType& normal_voxel =
                block->getVoxelByVoxelIndex(voxel_index);

            if (normal_voxel.small_type_status.is_this_type) {
              extractMeshFromSubvoxelInsideNormalVoxel(
                  normal_voxel, coords, true, &next_mesh_index, mesh.get());

              if (normal_voxel.small_type_status.is_this_type) {
                extractMeshFromSubvoxelOnNormalVoxelBorder(
                    voxel_index, block, coords, true, &next_mesh_index,
                    mesh.get());
              }
            }
            if (normal_voxel.middle_type_status.is_this_type) {
              extractMeshFromSubvoxelInsideNormalVoxel(
                  normal_voxel, coords, false, &next_mesh_index, mesh.get());

              if (normal_voxel.middle_type_status.is_this_type) {
                extractMeshFromSubvoxelOnNormalVoxelBorder(
                    voxel_index, block, coords, false, &next_mesh_index,
                    mesh.get());
              }
            }
          }
        }
      }
    }
  }

  void extractBlockMeshAdaptive(typename Block<VoxelType>::ConstPtr block,
                                Mesh::Ptr mesh) {
    DCHECK(block != nullptr);
    DCHECK(mesh != nullptr);

    // (TODO: jianhao) This can be implemented whenever voxel size is updated
    cube_coord_offsets_large_ =
        cube_index_offsets_.cast<FloatingPoint>() * voxel_size_;
    cube_coord_offsets_middle_ =
        cube_coord_offsets_large_ /
        static_cast<FloatingPoint>(adaptive_ratio_middle);
    cube_coord_offsets_small_ =
        cube_coord_offsets_large_ /
        static_cast<FloatingPoint>(adaptive_ratio_small);

    IndexElement vps = block->voxels_per_side();
    VertexIndex next_mesh_index = 0;

    VoxelIndex voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < vps - 1; ++voxel_index.x()) {
      for (voxel_index.y() = 0; voxel_index.y() < vps - 1; ++voxel_index.y()) {
        for (voxel_index.z() = 0; voxel_index.z() < vps - 1;
             ++voxel_index.z()) {
          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
          extractMeshInsideBlockAdaptive(*block, voxel_index, coords,
                                         &next_mesh_index, mesh.get());
        }
      }
    }

    // Max X plane
    // takes care of edge (x_max, y_max, z),
    // takes care of edge (x_max, y, z_max).
    voxel_index.x() = vps - 1;
    for (voxel_index.z() = 0; voxel_index.z() < vps; voxel_index.z()++) {
      for (voxel_index.y() = 0; voxel_index.y() < vps; voxel_index.y()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorderAdaptive(*block, voxel_index, coords,
                                    &next_mesh_index, mesh.get());
      }
    }

    // Max Y plane.
    // takes care of edge (x, y_max, z_max),
    // without corner (x_max, y_max, z_max).
    voxel_index.y() = vps - 1;
    for (voxel_index.z() = 0; voxel_index.z() < vps; voxel_index.z()++) {
      for (voxel_index.x() = 0; voxel_index.x() < vps - 1; voxel_index.x()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorderAdaptive(*block, voxel_index, coords,
                                    &next_mesh_index, mesh.get());
      }
    }

    // Max Z plane.
    voxel_index.z() = vps - 1;
    for (voxel_index.y() = 0; voxel_index.y() < vps - 1; voxel_index.y()++) {
      for (voxel_index.x() = 0; voxel_index.x() < vps - 1; voxel_index.x()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorderAdaptive(*block, voxel_index, coords,
                                    &next_mesh_index, mesh.get());
      }
    }

    for (voxel_index.x() = 0; voxel_index.x() < vps; ++voxel_index.x()) {
      for (voxel_index.y() = 0; voxel_index.y() < vps; ++voxel_index.y()) {
        for (voxel_index.z() = 0; voxel_index.z() < vps; ++voxel_index.z()) {
          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
          const VoxelType& normal_voxel =
              block->getVoxelByVoxelIndex(voxel_index);

          if (normal_voxel.small_type_status.should_be_divided()) {
            extractMeshFromSubvoxelInsideNormalVoxel(
                normal_voxel, coords, true, &next_mesh_index, mesh.get());
          } else if (normal_voxel.middle_type_status.should_be_divided()) {
            extractMeshFromSubvoxelInsideNormalVoxel(
                normal_voxel, coords, false, &next_mesh_index, mesh.get());
          }
        }
      }
    }
  }

  void extractMeshInsideBlockAdaptive(const Block<VoxelType>& block,
                                      const VoxelIndex& index,
                                      const Point& coords,
                                      VertexIndex* next_mesh_index,
                                      Mesh* mesh) {
    DCHECK(next_mesh_index != nullptr);
    DCHECK(mesh != nullptr);

    Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets =
        cube_index_offsets_.cast<FloatingPoint>() * voxel_size_;
    Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
    Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
    bool all_neighbors_observed = true;
    std::vector<const VoxelType*> large_voxels;
    std::vector<int> resolution_levels;
    large_voxels.reserve(8);
    resolution_levels.reserve(8);
    bool all_large_voxels = true;

    for (unsigned int i = 0; i < 8; ++i) {
      VoxelIndex corner_index = index + cube_index_offsets_.col(i);
      const VoxelType& voxel = block.getVoxelByVoxelIndex(corner_index);

      if (!utils::getSdfIfValid(voxel, config_.min_weight, &(corner_sdf(i)))) {
        all_neighbors_observed = false;
        break;
      }

      corner_coords.col(i) = coords + cube_coord_offsets.col(i);

      large_voxels.push_back(block.getVoxelPtrByVoxelIndex(corner_index));
      if (voxel.small_type_status.should_be_divided()) {
        resolution_levels.push_back(3);
        all_large_voxels = false;
      } else if (voxel.middle_type_status.should_be_divided()) {
        resolution_levels.push_back(2);
        all_large_voxels = false;
      } else {
        resolution_levels.push_back(1);
      }
    }

    if (all_neighbors_observed) {
      if (all_large_voxels) {
        MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                mesh);
      } else {
        extractMeshBoundaryMultiResolutionCube(large_voxels, resolution_levels,
                                               coords, next_mesh_index, mesh);
      }
    }
  }

  void extractMeshOnBorderAdaptive(const Block<VoxelType>& block,
                                   const VoxelIndex& index, const Point& coords,
                                   VertexIndex* next_mesh_index, Mesh* mesh) {
    DCHECK(mesh != nullptr);

    Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets =
        cube_index_offsets_.cast<FloatingPoint>() * voxel_size_;
    Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
    Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
    bool all_neighbors_observed = true;
    corner_coords.setZero();
    corner_sdf.setZero();

    std::vector<const VoxelType*> large_voxels;
    std::vector<int> resolution_levels;
    large_voxels.reserve(8);
    resolution_levels.reserve(8);
    bool all_large_voxels = true;

    for (unsigned int i = 0; i < 8; ++i) {
      VoxelIndex corner_index = index + cube_index_offsets_.col(i);

      if (block.isValidVoxelIndex(corner_index)) {
        const VoxelType& voxel = block.getVoxelByVoxelIndex(corner_index);

        if (!utils::getSdfIfValid(voxel, config_.min_weight,
                                  &(corner_sdf(i)))) {
          all_neighbors_observed = false;
          break;
        }

        corner_coords.col(i) = coords + cube_coord_offsets.col(i);

        large_voxels.push_back(block.getVoxelPtrByVoxelIndex(corner_index));
        if (voxel.small_type_status.should_be_divided()) {
          resolution_levels.push_back(3);
          all_large_voxels = false;
        } else if (voxel.middle_type_status.should_be_divided()) {
          resolution_levels.push_back(2);
          all_large_voxels = false;
        } else {
          resolution_levels.push_back(1);
        }
      } else {
        // We have to access a different block.
        BlockIndex block_offset = BlockIndex::Zero();

        for (unsigned int j = 0u; j < 3u; j++) {
          if (corner_index(j) < 0) {
            block_offset(j) = -1;
            corner_index(j) = corner_index(j) + voxels_per_side_;
          } else if (corner_index(j) >=
                     static_cast<IndexElement>(voxels_per_side_)) {
            block_offset(j) = 1;
            corner_index(j) = corner_index(j) - voxels_per_side_;
          }
        }

        BlockIndex neighbor_index = block.block_index() + block_offset;

        if (sdf_layer_const_->hasBlock(neighbor_index)) {
          const Block<VoxelType>& neighbor_block =
              sdf_layer_const_->getBlockByIndex(neighbor_index);

          CHECK(neighbor_block.isValidVoxelIndex(corner_index));
          const VoxelType& voxel =
              neighbor_block.getVoxelByVoxelIndex(corner_index);

          if (!utils::getSdfIfValid(voxel, config_.min_weight,
                                    &(corner_sdf(i)))) {
            all_neighbors_observed = false;
            break;
          }

          corner_coords.col(i) = coords + cube_coord_offsets.col(i);
          large_voxels.push_back(
              neighbor_block.getVoxelPtrByVoxelIndex(corner_index));
          if (voxel.small_type_status.should_be_divided()) {
            resolution_levels.push_back(3);
            all_large_voxels = false;
          } else if (voxel.middle_type_status.should_be_divided()) {
            resolution_levels.push_back(2);
            all_large_voxels = false;
          } else {
            resolution_levels.push_back(1);
          }
        } else {
          all_neighbors_observed = false;
          break;
        }
      }
    }

    if (all_neighbors_observed) {
      if (all_large_voxels) {
        MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                mesh);
      } else {
        extractMeshBoundaryMultiResolutionCube(large_voxels, resolution_levels,
                                               coords, next_mesh_index, mesh);
      }
    }
  }

  bool ifAllSubVoxelInvalid(const VoxelType& large_voxel,
                            const int& voxel_resolution_level) {
    if (voxel_resolution_level == 2) {
      VoxelIndex voxel_idx;
      for (int i = 0; i < large_voxel.child_voxels_small.size(); ++i) {
        if (large_voxel.child_voxels_small[i].weight > config_.min_weight) {
          return false;
        }
      }
      return true;
    } else if (voxel_resolution_level == 3) {
      VoxelIndex voxel_idx;
      for (int i = 0; i < large_voxel.child_voxels_queried.size(); ++i) {
        if (large_voxel.child_voxels_queried[i].weight > config_.min_weight) {
          return false;
        }
      }
      return true;
    }
  }

  bool searchIfSubvoxelFaceAllZero(const VoxelType* large_voxel,
                                   const int& voxel_resolution_level,
                                   const int& which_axis,
                                   const int& value_on_the_axis) {
    if (which_axis == 0) {
      VoxelIndex voxel_idx;
      if (voxel_resolution_level == 2) {
        voxel_idx.x() = value_on_the_axis;
        for (voxel_idx.y() = 0; voxel_idx.y() < adaptive_ratio_middle;
             ++voxel_idx.y()) {
          for (voxel_idx.z() = 0; voxel_idx.z() < adaptive_ratio_middle;
               ++voxel_idx.z()) {
            if (large_voxel
                    ->child_voxels_small[voxel_idx.x() +
                                         adaptive_ratio_middle * voxel_idx.y() +
                                         adaptive_ratio_middle *
                                             adaptive_ratio_middle *
                                             voxel_idx.z()]
                    .weight > config_.min_weight) {
              return false;
            }
          }
        }
        return true;
      } else if (voxel_resolution_level == 3) {
        voxel_idx.x() = value_on_the_axis;
        for (voxel_idx.y() = 0; voxel_idx.y() < adaptive_ratio_small;
             ++voxel_idx.y()) {
          for (voxel_idx.z() = 0; voxel_idx.z() < adaptive_ratio_small;
               ++voxel_idx.z()) {
            if (large_voxel
                    ->child_voxels_queried
                        [voxel_idx.x() + adaptive_ratio_small * voxel_idx.y() +
                         adaptive_ratio_small * adaptive_ratio_small *
                             voxel_idx.z()]
                    .weight > config_.min_weight) {
              return false;
            }
          }
        }
        return true;
      }
    } else if (which_axis == 1) {
      VoxelIndex voxel_idx;
      if (voxel_resolution_level == 2) {
        voxel_idx.y() = value_on_the_axis;
        for (voxel_idx.x() = 0; voxel_idx.x() < adaptive_ratio_middle;
             ++voxel_idx.x()) {
          for (voxel_idx.z() = 0; voxel_idx.z() < adaptive_ratio_middle;
               ++voxel_idx.z()) {
            if (large_voxel
                    ->child_voxels_small[voxel_idx.x() +
                                         adaptive_ratio_middle * voxel_idx.y() +
                                         adaptive_ratio_middle *
                                             adaptive_ratio_middle *
                                             voxel_idx.z()]
                    .weight > config_.min_weight) {
              return false;
            }
          }
        }
        return true;
      } else if (voxel_resolution_level == 3) {
        voxel_idx.y() = value_on_the_axis;
        for (voxel_idx.x() = 0; voxel_idx.x() < adaptive_ratio_small;
             ++voxel_idx.x()) {
          for (voxel_idx.z() = 0; voxel_idx.z() < adaptive_ratio_small;
               ++voxel_idx.z()) {
            if (large_voxel
                    ->child_voxels_queried
                        [voxel_idx.x() + adaptive_ratio_small * voxel_idx.y() +
                         adaptive_ratio_small * adaptive_ratio_small *
                             voxel_idx.z()]
                    .weight > config_.min_weight) {
              return false;
            }
          }
        }
        return true;
      }
    } else if (which_axis == 2) {
      VoxelIndex voxel_idx;
      if (voxel_resolution_level == 2) {
        voxel_idx.z() = value_on_the_axis;
        for (voxel_idx.x() = 0; voxel_idx.x() < adaptive_ratio_middle;
             ++voxel_idx.x()) {
          for (voxel_idx.y() = 0; voxel_idx.y() < adaptive_ratio_middle;
               ++voxel_idx.y()) {
            if (large_voxel
                    ->child_voxels_small[voxel_idx.x() +
                                         adaptive_ratio_middle * voxel_idx.y() +
                                         adaptive_ratio_middle *
                                             adaptive_ratio_middle *
                                             voxel_idx.z()]
                    .weight > config_.min_weight) {
              return false;
            }
          }
        }
        return true;
      } else if (voxel_resolution_level == 3) {
        voxel_idx.z() = value_on_the_axis;
        for (voxel_idx.x() = 0; voxel_idx.x() < adaptive_ratio_small;
             ++voxel_idx.x()) {
          for (voxel_idx.y() = 0; voxel_idx.y() < adaptive_ratio_small;
               ++voxel_idx.y()) {
            if (large_voxel
                    ->child_voxels_queried
                        [voxel_idx.x() + adaptive_ratio_small * voxel_idx.y() +
                         adaptive_ratio_small * adaptive_ratio_small *
                             voxel_idx.z()]
                    .weight > config_.min_weight) {
              return false;
            }
          }
        }
        return true;
      }
    }
  }

  TsdfSubVoxel getSubVoxelAdaptive(
      const VoxelType* large_voxel, const int& voxel_resolution_level,
      const int& queried_resolution_level, const VoxelIndex& queried_index,
      const unsigned int& corner_idx,
      Eigen::Matrix<FloatingPoint, 3, 8>* corner_coords) {
    // This function will find the voxel in the finest resolution that contains
    // the queried subvoxels and update the corresponding coordinate into
    // corner_coords.

    // Important !!! : the corrdinate updated to corner_coords is the coordinate
    // inside the largest voxel. Don't forget to add the global coordinates of
    // left-bottom corner of the large voxel to retrieve the exact coordinate
    CHECK_GE(queried_resolution_level, voxel_resolution_level);
    CHECK_GT(queried_resolution_level, 1);
    TsdfSubVoxel sub_voxel;
    if (queried_resolution_level == 2) {
      if (voxel_resolution_level == queried_resolution_level) {
        sub_voxel =
            large_voxel->child_voxels_small
                [queried_index(0) + adaptive_ratio_middle * queried_index(1) +
                 adaptive_ratio_middle * adaptive_ratio_middle *
                     queried_index(2)];

        for (unsigned int i = 0; i < 3; ++i) {
          corner_coords->col(corner_idx)(i) =
              (static_cast<FloatingPoint>(queried_index(i)) + 0.5f) *
              voxel_size_ / static_cast<FloatingPoint>(adaptive_ratio_middle);
        }

      } else {
        // we can only use the large_voxel itself. In mesh generation, only sdf
        // and weight is required
        sub_voxel.weight = large_voxel->weight;
        sub_voxel.distance = large_voxel->distance;

        for (unsigned int i = 0; i < 3; ++i) {
          corner_coords->col(corner_idx)(i) = 0.5f * voxel_size_;
        }
      }
    } else if (queried_resolution_level == 3) {
      if (voxel_resolution_level == queried_resolution_level) {
        sub_voxel =
            large_voxel->child_voxels_queried
                [queried_index(0) + adaptive_ratio_small * queried_index(1) +
                 adaptive_ratio_small * adaptive_ratio_small *
                     queried_index(2)];

        for (unsigned int i = 0; i < 3; ++i) {
          corner_coords->col(corner_idx)(i) =
              (static_cast<FloatingPoint>(queried_index(i)) + 0.5f) *
              voxel_size_ / static_cast<FloatingPoint>(adaptive_ratio_small);
        }

      } else if (voxel_resolution_level == 2) {
        int ratio = adaptive_ratio_small / adaptive_ratio_middle;
        sub_voxel =
            large_voxel->child_voxels_small[queried_index(0) / ratio +
                                            adaptive_ratio_middle *
                                                (queried_index(1) / ratio) +
                                            adaptive_ratio_middle *
                                                adaptive_ratio_middle *
                                                (queried_index(2) / ratio)];

        for (unsigned int i = 0; i < 3; ++i) {
          corner_coords->col(corner_idx)(i) =
              (static_cast<FloatingPoint>(queried_index(i) / ratio) + 0.5f) *
              voxel_size_ / static_cast<FloatingPoint>(adaptive_ratio_middle);
        }

      } else {
        // we can only use the large_voxel itself. In mesh generation, only sdf
        // and weight is required
        sub_voxel.weight = large_voxel->weight;
        sub_voxel.distance = large_voxel->distance;

        for (unsigned int i = 0; i < 3; ++i) {
          corner_coords->col(corner_idx)(i) = 0.5f * voxel_size_;
        }
      }
    }

    return sub_voxel;
  }

  void extractMeshBoundaryMultiResolutionCube(
      const std::vector<const VoxelType*>& large_voxels,
      std::vector<int>& resolution_levels, const Point& coords,
      VertexIndex* next_mesh_index, Mesh* mesh) {
    CHECK_EQ(large_voxels.size(), 8);
    CHECK_EQ(resolution_levels.size(), 8);
    Point origin = coords;
    origin.x() -= 0.5f * voxel_size_;
    origin.y() -= 0.5f * voxel_size_;
    origin.z() -= 0.5f * voxel_size_;

    // 12 faces
    unsigned int large_voxel_1_idx;
    unsigned int large_voxel_2_idx;
    Point origin_voxel_1;

    for (int face_idx = 0; face_idx < 12; ++face_idx) {
      large_voxel_1_idx = cube_index_face_(0, face_idx);
      // if (large_voxel_1_idx != 0) continue;
      large_voxel_2_idx = cube_index_face_(1, face_idx);

      for (unsigned int i = 0; i < 3; ++i) {
        origin_voxel_1(i) =
            origin(i) + static_cast<FloatingPoint>(
                            cube_index_offsets_(i, large_voxel_1_idx)) *
                            voxel_size_;
      }

      extractMeshBoundaryMultiResolutionFace(
          large_voxels[large_voxel_1_idx], large_voxels[large_voxel_2_idx],
          resolution_levels[large_voxel_1_idx],
          resolution_levels[large_voxel_2_idx], axis_index_face_(0, face_idx),
          axis_index_face_(1, face_idx), axis_index_face_(2, face_idx),
          origin_voxel_1, next_mesh_index, mesh);
    }

    // 6 edges
    unsigned int large_voxel_3_idx;
    unsigned int large_voxel_4_idx;
    for (int edge_idx = 0; edge_idx < 6; ++edge_idx) {
      large_voxel_1_idx = cube_index_edge_(0, edge_idx);
      // if (large_voxel_1_idx != 0) continue;
      large_voxel_2_idx = cube_index_edge_(1, edge_idx);
      large_voxel_3_idx = cube_index_edge_(2, edge_idx);
      large_voxel_4_idx = cube_index_edge_(3, edge_idx);

      for (unsigned int i = 0; i < 3; ++i) {
        origin_voxel_1(i) =
            origin(i) + static_cast<FloatingPoint>(
                            cube_index_offsets_(i, large_voxel_1_idx)) *
                            voxel_size_;
      }

      extractMeshBoundaryMultiResolutionEdge(
          large_voxels[large_voxel_1_idx], large_voxels[large_voxel_2_idx],
          large_voxels[large_voxel_3_idx], large_voxels[large_voxel_4_idx],
          resolution_levels[large_voxel_1_idx],
          resolution_levels[large_voxel_2_idx],
          resolution_levels[large_voxel_3_idx],
          resolution_levels[large_voxel_4_idx], axis_index_edge_(0, edge_idx),
          axis_index_edge_(1, edge_idx), axis_index_edge_(2, edge_idx),
          origin_voxel_1, next_mesh_index, mesh);
    }

    // 1 corner
    {
      int this_resolution_level = std::max(
          std::max(std::max(resolution_levels[0], resolution_levels[1]),
                   std::max(resolution_levels[2], resolution_levels[3])),
          std::max(std::max(resolution_levels[4], resolution_levels[5]),
                   std::max(resolution_levels[6], resolution_levels[7])));

      Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
      Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
      bool all_neighbors_observed = true;

      if (this_resolution_level == 2) {
        VoxelIndex sub_idx;
        for (unsigned int i = 0; i < 8; ++i) {
          for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
            if (cube_index_offsets_(axis_idx, i) == 0) {
              sub_idx(axis_idx) = adaptive_ratio_middle - 1;
            } else {
              sub_idx(axis_idx) = 0;
            }
          }
          // sub_idx = sub_voxel_index_offsets.col(i);

          TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
              large_voxels[i], resolution_levels[i], this_resolution_level,
              sub_idx, i, &corner_coords);
          for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
            corner_coords(axis_idx, i) +=
                origin(axis_idx) +
                static_cast<FloatingPoint>(cube_index_offsets_(axis_idx, i)) *
                    voxel_size_;
          }
          const TsdfSubVoxel const_sub_voxel = sub_voxel;
          if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                    &(corner_sdf(i)))) {
            all_neighbors_observed = false;
            break;
          }
        }

        if (all_neighbors_observed) {
          MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                  mesh);
        }
      } else if (this_resolution_level == 3) {
        VoxelIndex sub_idx;
        for (unsigned int i = 0; i < 8; ++i) {
          for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
            if (cube_index_offsets_(axis_idx, i) == 0) {
              sub_idx(axis_idx) = adaptive_ratio_small - 1;
            } else {
              sub_idx(axis_idx) = 0;
            }
          }

          TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
              large_voxels[i], resolution_levels[i], this_resolution_level,
              sub_idx, i, &corner_coords);
          for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
            corner_coords(axis_idx, i) +=
                origin(axis_idx) +
                static_cast<FloatingPoint>(cube_index_offsets_(axis_idx, i)) *
                    voxel_size_;
          }
          const TsdfSubVoxel const_sub_voxel = sub_voxel;
          if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                    &(corner_sdf(i)))) {
            all_neighbors_observed = false;
            break;
          }
        }

        if (all_neighbors_observed) {
          MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                  mesh);
        }
      }
    }
  }

  void extractMeshBoundaryMultiResolutionFace(
      const VoxelType* large_voxel_1, const VoxelType* large_voxel_2,
      const int& resolution_level_1, const int& resolution_level_2,
      const int& across_axis, const int& face_axis1, const int& face_axis2,
      const Point& origin_voxel_1, VertexIndex* next_mesh_index, Mesh* mesh) {
    int this_resolution_level =
        std::max(resolution_level_1, resolution_level_2);
    bool all_neighbors_observed;

    if (this_resolution_level == 2) {
      VoxelIndex sub_idx_1;

      sub_idx_1(across_axis) = adaptive_ratio_middle - 1;
      for (sub_idx_1(face_axis1) = 0;
           sub_idx_1(face_axis1) < adaptive_ratio_middle - 1;
           ++sub_idx_1(face_axis1)) {
        for (sub_idx_1(face_axis2) = 0;
             sub_idx_1(face_axis2) < adaptive_ratio_middle - 1;
             ++sub_idx_1(face_axis2)) {
          Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
          Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
          all_neighbors_observed = true;

          for (unsigned int i = 0; i < 8; ++i) {
            // the idx of sub_voxels in the largest voxel
            VoxelIndex sub_idx;
            sub_idx = sub_idx_1;
            sub_idx(face_axis1) += cube_index_offsets_(face_axis1, i);
            sub_idx(face_axis2) += cube_index_offsets_(face_axis2, i);
            if (cube_index_offsets_(across_axis, i) == 0) {
              sub_idx(across_axis) = adaptive_ratio_middle - 1;
              TsdfSubVoxel sub_voxel;
              sub_voxel = getSubVoxelAdaptive(large_voxel_1, resolution_level_1,
                                              this_resolution_level, sub_idx, i,
                                              &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            } else {
              // the across_axis of sub_idx is always fixed, it's 0 for the
              // second large voxel
              sub_idx(across_axis) = 0;
              TsdfSubVoxel sub_voxel;
              sub_voxel = getSubVoxelAdaptive(large_voxel_2, resolution_level_2,
                                              this_resolution_level, sub_idx, i,
                                              &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              corner_coords(across_axis, i) += voxel_size_;
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            }
          }

          if (all_neighbors_observed) {
            MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                    mesh);
          }
        }
      }
    } else if (this_resolution_level == 3) {
      VoxelIndex sub_idx_1;

      sub_idx_1(across_axis) = adaptive_ratio_small - 1;
      for (sub_idx_1(face_axis1) = 0;
           sub_idx_1(face_axis1) < adaptive_ratio_small - 1;
           ++sub_idx_1(face_axis1)) {
        for (sub_idx_1(face_axis2) = 0;
             sub_idx_1(face_axis2) < adaptive_ratio_small - 1;
             ++sub_idx_1(face_axis2)) {
          Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
          Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
          all_neighbors_observed = true;

          for (unsigned int i = 0; i < 8; ++i) {
            // the idx of sub_voxels in the largest voxel
            VoxelIndex sub_idx;
            sub_idx = sub_idx_1;
            sub_idx(face_axis1) += cube_index_offsets_(face_axis1, i);
            sub_idx(face_axis2) += cube_index_offsets_(face_axis2, i);
            if (cube_index_offsets_(across_axis, i) == 0) {
              sub_idx(across_axis) = adaptive_ratio_small - 1;
              TsdfSubVoxel sub_voxel;
              sub_voxel = getSubVoxelAdaptive(large_voxel_1, resolution_level_1,
                                              this_resolution_level, sub_idx, i,
                                              &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            } else {
              // the across_axis of sub_idx is always fixed, it's 0 for the
              // second large voxel
              sub_idx(across_axis) = 0;
              TsdfSubVoxel sub_voxel;
              sub_voxel = getSubVoxelAdaptive(large_voxel_2, resolution_level_2,
                                              this_resolution_level, sub_idx, i,
                                              &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              corner_coords(across_axis, i) += voxel_size_;
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            }
          }

          if (all_neighbors_observed) {
            MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                    mesh);
          }
        }
      }
    }
  }

  void extractMeshBoundaryMultiResolutionEdge(
      const VoxelType* large_voxel_1, const VoxelType* large_voxel_2,
      const VoxelType* large_voxel_3, const VoxelType* large_voxel_4,
      const int& resolution_level_1, const int& resolution_level_2,
      const int& resolution_level_3, const int& resolution_level_4,
      const int& other_axis1, const int& other_axis2, const int& edge_axis,
      const Point& origin_voxel_1, VertexIndex* next_mesh_index, Mesh* mesh) {
    int this_resolution_level =
        std::max(std::max(resolution_level_1, resolution_level_2),
                 std::max(resolution_level_3, resolution_level_4));
    bool all_neighbors_observed;

    if (this_resolution_level == 2) {
      VoxelIndex sub_idx_1;

      sub_idx_1(other_axis1) = adaptive_ratio_middle - 1;
      sub_idx_1(other_axis2) = adaptive_ratio_middle - 1;
      for (sub_idx_1(edge_axis) = 0;
           sub_idx_1(edge_axis) < adaptive_ratio_middle - 1;
           ++sub_idx_1(edge_axis)) {
        Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
        Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
        all_neighbors_observed = true;

        for (unsigned int i = 0; i < 8; ++i) {
          // the idx of sub_voxels in the largest voxel
          VoxelIndex sub_idx;
          sub_idx = sub_idx_1;
          sub_idx(edge_axis) += cube_index_offsets_(edge_axis, i);
          if (cube_index_offsets_(other_axis1, i) == 0) {
            if (cube_index_offsets_(other_axis2, i) == 0) {
              sub_idx(other_axis1) = adaptive_ratio_middle - 1;
              sub_idx(other_axis2) = adaptive_ratio_middle - 1;
              TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
                  large_voxel_1, resolution_level_1, this_resolution_level,
                  sub_idx, i, &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            } else {
              sub_idx(other_axis1) = adaptive_ratio_middle - 1;
              sub_idx(other_axis2) = 0;
              TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
                  large_voxel_2, resolution_level_2, this_resolution_level,
                  sub_idx, i, &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              corner_coords(other_axis2, i) += voxel_size_;
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            }
          } else {
            if (cube_index_offsets_(other_axis2, i) == 0) {
              sub_idx(other_axis1) = 0;
              sub_idx(other_axis2) = adaptive_ratio_middle - 1;
              TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
                  large_voxel_3, resolution_level_3, this_resolution_level,
                  sub_idx, i, &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              corner_coords(other_axis1, i) += voxel_size_;
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            } else {
              sub_idx(other_axis1) = 0;
              sub_idx(other_axis2) = 0;
              TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
                  large_voxel_4, resolution_level_4, this_resolution_level,
                  sub_idx, i, &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              corner_coords(other_axis1, i) += voxel_size_;
              corner_coords(other_axis2, i) += voxel_size_;
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            }
          }
        }

        if (all_neighbors_observed) {
          MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                  mesh);
        }
      }
    } else if (this_resolution_level == 3) {
      VoxelIndex sub_idx_1;

      sub_idx_1(other_axis1) = adaptive_ratio_small - 1;
      sub_idx_1(other_axis2) = adaptive_ratio_small - 1;
      for (sub_idx_1(edge_axis) = 0;
           sub_idx_1(edge_axis) < adaptive_ratio_small - 1;
           ++sub_idx_1(edge_axis)) {
        Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
        Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
        all_neighbors_observed = true;

        for (unsigned int i = 0; i < 8; ++i) {
          // the idx of sub_voxels in the largest voxel
          VoxelIndex sub_idx;
          sub_idx = sub_idx_1;
          sub_idx(edge_axis) += cube_index_offsets_(edge_axis, i);
          if (cube_index_offsets_(other_axis1, i) == 0) {
            if (cube_index_offsets_(other_axis2, i) == 0) {
              sub_idx(other_axis1) = adaptive_ratio_small - 1;
              sub_idx(other_axis2) = adaptive_ratio_small - 1;
              TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
                  large_voxel_1, resolution_level_1, this_resolution_level,
                  sub_idx, i, &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            } else {
              sub_idx(other_axis1) = adaptive_ratio_small - 1;
              sub_idx(other_axis2) = 0;

              TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
                  large_voxel_2, resolution_level_2, this_resolution_level,
                  sub_idx, i, &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              corner_coords(other_axis2, i) += voxel_size_;
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            }
          } else {
            if (cube_index_offsets_(other_axis2, i) == 0) {
              sub_idx(other_axis1) = 0;
              sub_idx(other_axis2) = adaptive_ratio_small - 1;
              TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
                  large_voxel_3, resolution_level_3, this_resolution_level,
                  sub_idx, i, &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              corner_coords(other_axis1, i) += voxel_size_;
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            } else {
              sub_idx(other_axis1) = 0;
              sub_idx(other_axis2) = 0;
              TsdfSubVoxel sub_voxel = getSubVoxelAdaptive(
                  large_voxel_4, resolution_level_4, this_resolution_level,
                  sub_idx, i, &corner_coords);
              for (unsigned int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                corner_coords(axis_idx, i) += origin_voxel_1(axis_idx);
              }
              corner_coords(other_axis1, i) += voxel_size_;
              corner_coords(other_axis2, i) += voxel_size_;
              const TsdfSubVoxel const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }
            }
          }
        }

        if (all_neighbors_observed) {
          MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                  mesh);
        }
      }
    }
  }

  void extractBlockMeshSelective(typename Block<VoxelType>::ConstPtr block,
                                 Mesh::Ptr mesh, const int& type_id) {
    DCHECK(block != nullptr);
    DCHECK(mesh != nullptr);

    IndexElement vps = block->voxels_per_side();
    VertexIndex next_mesh_index = 0;

    if (type_id == 1) {
      VoxelIndex voxel_index;
      for (voxel_index.x() = 0; voxel_index.x() < vps - 1; ++voxel_index.x()) {
        for (voxel_index.y() = 0; voxel_index.y() < vps - 1;
             ++voxel_index.y()) {
          for (voxel_index.z() = 0; voxel_index.z() < vps - 1;
               ++voxel_index.z()) {
            Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
            extractMeshInsideBlock(*block, voxel_index, coords,
                                   &next_mesh_index, mesh.get());
          }
        }
      }

      // Max X plane
      // takes care of edge (x_max, y_max, z),
      // takes care of edge (x_max, y, z_max).
      voxel_index.x() = vps - 1;
      for (voxel_index.z() = 0; voxel_index.z() < vps; voxel_index.z()++) {
        for (voxel_index.y() = 0; voxel_index.y() < vps; voxel_index.y()++) {
          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
          extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                              mesh.get());
        }
      }

      // Max Y plane.
      // takes care of edge (x, y_max, z_max),
      // without corner (x_max, y_max, z_max).
      voxel_index.y() = vps - 1;
      for (voxel_index.z() = 0; voxel_index.z() < vps; voxel_index.z()++) {
        for (voxel_index.x() = 0; voxel_index.x() < vps - 1;
             voxel_index.x()++) {
          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
          extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                              mesh.get());
        }
      }

      // Max Z plane.
      voxel_index.z() = vps - 1;
      for (voxel_index.y() = 0; voxel_index.y() < vps - 1; voxel_index.y()++) {
        for (voxel_index.x() = 0; voxel_index.x() < vps - 1;
             voxel_index.x()++) {
          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
          extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                              mesh.get());
        }
      }
    } else if (type_id == 2) {
      VoxelIndex voxel_index;
      for (voxel_index.x() = 0; voxel_index.x() < vps; ++voxel_index.x()) {
        for (voxel_index.y() = 0; voxel_index.y() < vps; ++voxel_index.y()) {
          for (voxel_index.z() = 0; voxel_index.z() < vps; ++voxel_index.z()) {
            Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
            const VoxelType& normal_voxel =
                block->getVoxelByVoxelIndex(voxel_index);

            if (normal_voxel.middle_type_status.is_this_type) {
              extractMeshFromSubvoxelInsideNormalVoxel(
                  normal_voxel, coords, false, &next_mesh_index, mesh.get());

              if (normal_voxel.middle_type_status.is_this_type) {
                extractMeshFromSubvoxelOnNormalVoxelBorder(
                    voxel_index, block, coords, false, &next_mesh_index,
                    mesh.get());
              }
            }
          }
        }
      }
    } else if (type_id == 3) {
      VoxelIndex voxel_index;
      for (voxel_index.x() = 0; voxel_index.x() < vps; ++voxel_index.x()) {
        for (voxel_index.y() = 0; voxel_index.y() < vps; ++voxel_index.y()) {
          for (voxel_index.z() = 0; voxel_index.z() < vps; ++voxel_index.z()) {
            Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
            const VoxelType& normal_voxel =
                block->getVoxelByVoxelIndex(voxel_index);

            if (normal_voxel.small_type_status.is_this_type) {
              extractMeshFromSubvoxelInsideNormalVoxel(
                  normal_voxel, coords, true, &next_mesh_index, mesh.get());

              if (normal_voxel.small_type_status.is_this_type) {
                extractMeshFromSubvoxelOnNormalVoxelBorder(
                    voxel_index, block, coords, true, &next_mesh_index,
                    mesh.get());
              }
            }
          }
        }
      }
    }
  }

  virtual void updateMeshForBlock(const BlockIndex& block_index) {
    Mesh::Ptr mesh = mesh_layer_->getMeshPtrByIndex(block_index);
    mesh->clear();
    // This block should already exist, otherwise it makes no sense to update
    // the mesh for it. ;)
    typename Block<VoxelType>::ConstPtr block =
        sdf_layer_const_->getBlockPtrByIndex(block_index);

    if (!block) {
      LOG(ERROR) << "Trying to mesh a non-existent block at index: "
                 << block_index.transpose();
      return;
    }

    if (adaptive_mapping) {
      extractBlockMeshAdaptive(block, mesh);
    } else {
      extractBlockMesh(block, mesh);
    }
    // Update colors if needed.
    if (config_.use_color) {
      updateMeshColor(*block, mesh.get());
    }

    mesh->updated = true;
  }

  virtual void updateMeshForBlockSelective(const BlockIndex& block_index,
                                           const int& type_id) {
    Mesh::Ptr mesh = mesh_layer_->getMeshPtrByIndex(block_index);
    mesh->clear();
    // This block should already exist, otherwise it makes no sense to update
    // the mesh for it. ;)
    typename Block<VoxelType>::ConstPtr block =
        sdf_layer_const_->getBlockPtrByIndex(block_index);

    if (!block) {
      LOG(ERROR) << "Trying to mesh a non-existent block at index: "
                 << block_index.transpose();
      return;
    }
    extractBlockMeshSelective(block, mesh, type_id);
    // Update colors if needed.
    if (config_.use_color) {
      updateMeshColor(*block, mesh.get());
    }

    mesh->updated = true;
  }

  virtual void updateMeshForBlockSemantic(
      const BlockIndex& block_index,
      const std::shared_ptr<voxblox::ColorMap>& color_map) {
    Mesh::Ptr mesh = mesh_layer_->getMeshPtrByIndex(block_index);
    mesh->clear();
    // This block should already exist, otherwise it makes no sense to update
    // the mesh for it. ;)
    typename Block<VoxelType>::ConstPtr block =
        sdf_layer_const_->getBlockPtrByIndex(block_index);

    if (!block) {
      LOG(ERROR) << "Trying to mesh a non-existent block at index: "
                 << block_index.transpose();
      return;
    }
    if (adaptive_mapping) {
      extractBlockMeshAdaptive(block, mesh);
    } else {
      extractBlockMesh(block, mesh);
    }
    // Update colors if needed.
    if (config_.use_color) {
      updateMeshColorSemantic(*block, mesh.get(), color_map);
    }

    mesh->updated = true;
  }

  void extractMeshInsideBlock(const Block<VoxelType>& block,
                              const VoxelIndex& index, const Point& coords,
                              VertexIndex* next_mesh_index, Mesh* mesh) {
    DCHECK(next_mesh_index != nullptr);
    DCHECK(mesh != nullptr);

    Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets =
        cube_index_offsets_.cast<FloatingPoint>() * voxel_size_;
    Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
    Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
    Eigen::Matrix<bool, 8, 1> not_large_semantic;
    bool all_neighbors_observed = true;

    for (unsigned int i = 0; i < 8; ++i) {
      VoxelIndex corner_index = index + cube_index_offsets_.col(i);
      const VoxelType& voxel = block.getVoxelByVoxelIndex(corner_index);

      if (!utils::getSdfIfValid(voxel, config_.min_weight, &(corner_sdf(i)))) {
        all_neighbors_observed = false;
        break;
      }

      corner_coords.col(i) = coords + cube_coord_offsets.col(i);

      if (adaptive_mapping) {
        not_large_semantic(i) = (voxel.small_type_status.is_this_type) ||
                                (voxel.middle_type_status.is_this_type);
      }
    }

    if (all_neighbors_observed) {
      if (adaptive_mapping) {
        MarchingCubes::meshCubeAdaptive(corner_coords, corner_sdf,
                                        not_large_semantic, next_mesh_index,
                                        mesh);
      } else {
        MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                mesh);
      }
    }
  }

  void extractMeshOnBorder(const Block<VoxelType>& block,
                           const VoxelIndex& index, const Point& coords,
                           VertexIndex* next_mesh_index, Mesh* mesh) {
    DCHECK(mesh != nullptr);

    Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets =
        cube_index_offsets_.cast<FloatingPoint>() * voxel_size_;
    Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
    Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
    bool all_neighbors_observed = true;
    corner_coords.setZero();
    corner_sdf.setZero();
    Eigen::Matrix<bool, 8, 1> not_large_semantic;

    for (unsigned int i = 0; i < 8; ++i) {
      VoxelIndex corner_index = index + cube_index_offsets_.col(i);

      if (block.isValidVoxelIndex(corner_index)) {
        const VoxelType& voxel = block.getVoxelByVoxelIndex(corner_index);

        if (!utils::getSdfIfValid(voxel, config_.min_weight,
                                  &(corner_sdf(i)))) {
          all_neighbors_observed = false;
          break;
        }

        corner_coords.col(i) = coords + cube_coord_offsets.col(i);

        if (adaptive_mapping) {
          not_large_semantic(i) = (voxel.small_type_status.is_this_type) ||
                                  (voxel.middle_type_status.is_this_type);
        }
      } else {
        // We have to access a different block.
        BlockIndex block_offset = BlockIndex::Zero();

        for (unsigned int j = 0u; j < 3u; j++) {
          if (corner_index(j) < 0) {
            block_offset(j) = -1;
            corner_index(j) = corner_index(j) + voxels_per_side_;
          } else if (corner_index(j) >=
                     static_cast<IndexElement>(voxels_per_side_)) {
            block_offset(j) = 1;
            corner_index(j) = corner_index(j) - voxels_per_side_;
          }
        }

        BlockIndex neighbor_index = block.block_index() + block_offset;

        if (sdf_layer_const_->hasBlock(neighbor_index)) {
          const Block<VoxelType>& neighbor_block =
              sdf_layer_const_->getBlockByIndex(neighbor_index);

          CHECK(neighbor_block.isValidVoxelIndex(corner_index));
          const VoxelType& voxel =
              neighbor_block.getVoxelByVoxelIndex(corner_index);

          if (!utils::getSdfIfValid(voxel, config_.min_weight,
                                    &(corner_sdf(i)))) {
            all_neighbors_observed = false;
            break;
          }

          corner_coords.col(i) = coords + cube_coord_offsets.col(i);
          if (adaptive_mapping) {
            not_large_semantic(i) = (voxel.small_type_status.is_this_type) ||
                                    (voxel.middle_type_status.is_this_type);
          }
        } else {
          all_neighbors_observed = false;
          break;
        }
      }
    }

    if (all_neighbors_observed) {
      if (adaptive_mapping) {
        MarchingCubes::meshCubeAdaptive(corner_coords, corner_sdf,
                                        not_large_semantic, next_mesh_index,
                                        mesh);
      } else {
        MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                mesh);
      }
    }
  }

  void extractMeshFromSubvoxelInsideNormalVoxel(
      const VoxelType& normal_voxel, const Point& normal_voxel_coords,
      const bool& extract_from_queried, VertexIndex* next_mesh_index,
      Mesh* mesh) {
    int voxel_size_ratio = 0;

    if (extract_from_queried) {
      // we are extracting mesh from the subvoxel of small semantic voxels
      CHECK(normal_voxel.small_type_status.should_be_divided());
      voxel_size_ratio = adaptive_ratio_small;
    } else {
      // TODO(jianhao) since we only have two levels for now, if it's not
      // queried semantic, then it's middle semantic
      CHECK(normal_voxel.middle_type_status.should_be_divided());
      voxel_size_ratio = adaptive_ratio_middle;
    }

    Point coords_first_subvoxel;
    coords_first_subvoxel(0) =
        normal_voxel_coords.x() - voxel_size_ / 2.0 +
        0.5 * voxel_size_ / static_cast<FloatingPoint>(voxel_size_ratio);
    coords_first_subvoxel(1) =
        normal_voxel_coords.y() - voxel_size_ / 2.0 +
        0.5 * voxel_size_ / static_cast<FloatingPoint>(voxel_size_ratio);
    coords_first_subvoxel(2) =
        normal_voxel_coords.z() - voxel_size_ / 2.0 +
        0.5 * voxel_size_ / static_cast<FloatingPoint>(voxel_size_ratio);

    Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets_subvoxel =
        cube_index_offsets_.cast<FloatingPoint>() * voxel_size_ /
        static_cast<FloatingPoint>(voxel_size_ratio);
    VoxelIndex sub_voxel_index;

    for (sub_voxel_index.x() = 0; sub_voxel_index.x() < voxel_size_ratio - 1;
         ++sub_voxel_index.x()) {
      for (sub_voxel_index.y() = 0; sub_voxel_index.y() < voxel_size_ratio - 1;
           ++sub_voxel_index.y()) {
        for (sub_voxel_index.z() = 0;
             sub_voxel_index.z() < voxel_size_ratio - 1;
             ++sub_voxel_index.z()) {
          Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
          Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
          bool all_neighbors_observed = true;
          corner_coords.setZero();
          corner_sdf.setZero();

          Point coords_current_subvoxel;
          coords_current_subvoxel(0) =
              coords_first_subvoxel.x() +
              static_cast<FloatingPoint>(sub_voxel_index.x()) * voxel_size_ /
                  static_cast<FloatingPoint>(voxel_size_ratio);
          coords_current_subvoxel(1) =
              coords_first_subvoxel.y() +
              static_cast<FloatingPoint>(sub_voxel_index.y()) * voxel_size_ /
                  static_cast<FloatingPoint>(voxel_size_ratio);
          coords_current_subvoxel(2) =
              coords_first_subvoxel.z() +
              static_cast<FloatingPoint>(sub_voxel_index.z()) * voxel_size_ /
                  static_cast<FloatingPoint>(voxel_size_ratio);

          for (unsigned int i = 0; i < 8; ++i) {
            VoxelIndex corner_index =
                sub_voxel_index + cube_index_offsets_.col(i);

            TsdfSubVoxel sub_voxel;
            if (extract_from_queried) {
              sub_voxel =
                  normal_voxel.child_voxels_queried
                      [corner_index(0) + voxel_size_ratio * corner_index(1) +
                       voxel_size_ratio * voxel_size_ratio * corner_index(2)];
            } else {
              sub_voxel =
                  normal_voxel.child_voxels_small
                      [corner_index(0) + voxel_size_ratio * corner_index(1) +
                       voxel_size_ratio * voxel_size_ratio * corner_index(2)];
            }

            const TsdfSubVoxel& const_sub_voxel = sub_voxel;
            if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                      &(corner_sdf(i)))) {
              all_neighbors_observed = false;
              break;
            }

            corner_coords.col(i) =
                coords_current_subvoxel + cube_coord_offsets_subvoxel.col(i);
          }

          if (all_neighbors_observed) {
            MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                    mesh);
          }
        }
      }
    }
  }

  void extractMeshFromSubvoxelOnNormalVoxelBorder(
      const VoxelIndex& normal_voxel_index,
      typename Block<VoxelType>::ConstPtr block,
      const Point& normal_voxel_coords, const bool& extract_from_queried,
      VertexIndex* next_mesh_index, Mesh* mesh) {
    const VoxelType& normal_voxel =
        block->getVoxelByVoxelIndex(normal_voxel_index);

    int voxel_size_ratio = 0;

    if (extract_from_queried) {
      // we are extracting mesh from the subvoxel of small semantic voxels
      CHECK(normal_voxel.small_type_status.should_be_divided());
      voxel_size_ratio = adaptive_ratio_small;
    } else {
      // TODO(jianhao) since we only have two levels for now, if it's not
      // queried semantic, then it's middle semantic
      CHECK(normal_voxel.middle_type_status.should_be_divided());
      voxel_size_ratio = adaptive_ratio_middle;
    }

    Point coords_first_subvoxel;
    coords_first_subvoxel(0) =
        normal_voxel_coords.x() - voxel_size_ / 2.0 +
        0.5 * voxel_size_ / static_cast<FloatingPoint>(voxel_size_ratio);
    coords_first_subvoxel(1) =
        normal_voxel_coords.y() - voxel_size_ / 2.0 +
        0.5 * voxel_size_ / static_cast<FloatingPoint>(voxel_size_ratio);
    coords_first_subvoxel(2) =
        normal_voxel_coords.z() - voxel_size_ / 2.0 +
        0.5 * voxel_size_ / static_cast<FloatingPoint>(voxel_size_ratio);

    Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets_subvoxel =
        cube_index_offsets_.cast<FloatingPoint>() * voxel_size_ /
        static_cast<FloatingPoint>(voxel_size_ratio);
    VoxelIndex sub_voxel_index;

    for (sub_voxel_index.x() = 0; sub_voxel_index.x() < voxel_size_ratio;
         ++sub_voxel_index.x()) {
      for (sub_voxel_index.y() = 0; sub_voxel_index.y() < voxel_size_ratio;
           ++sub_voxel_index.y()) {
        for (sub_voxel_index.z() = 0; sub_voxel_index.z() < voxel_size_ratio;
             ++sub_voxel_index.z()) {
          // we only process subvoxel on borders
          if (sub_voxel_index.x() < voxel_size_ratio - 1 &&
              sub_voxel_index.y() < voxel_size_ratio - 1 &&
              sub_voxel_index.z() < voxel_size_ratio - 1)
            continue;

          Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
          Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
          bool all_neighbors_observed = true;
          corner_coords.setZero();
          corner_sdf.setZero();

          Point coords_current_subvoxel;
          coords_current_subvoxel(0) =
              coords_first_subvoxel.x() +
              static_cast<FloatingPoint>(sub_voxel_index.x()) * voxel_size_ /
                  static_cast<FloatingPoint>(voxel_size_ratio);
          coords_current_subvoxel(1) =
              coords_first_subvoxel.y() +
              static_cast<FloatingPoint>(sub_voxel_index.y()) * voxel_size_ /
                  static_cast<FloatingPoint>(voxel_size_ratio);
          coords_current_subvoxel(2) =
              coords_first_subvoxel.z() +
              static_cast<FloatingPoint>(sub_voxel_index.z()) * voxel_size_ /
                  static_cast<FloatingPoint>(voxel_size_ratio);

          for (unsigned int i = 0; i < 8; ++i) {
            VoxelIndex corner_index =
                sub_voxel_index + cube_index_offsets_.col(i);

            if (corner_index.x() < voxel_size_ratio &&
                corner_index.y() < voxel_size_ratio &&
                corner_index.z() < voxel_size_ratio) {
              DCHECK_GE(corner_index.x(), 0);
              DCHECK_GE(corner_index.y(), 0);
              DCHECK_GE(corner_index.z(), 0);

              TsdfSubVoxel sub_voxel;
              if (extract_from_queried) {
                sub_voxel =
                    normal_voxel.child_voxels_queried
                        [corner_index(0) + voxel_size_ratio * corner_index(1) +
                         voxel_size_ratio * voxel_size_ratio * corner_index(2)];
              } else {
                sub_voxel =
                    normal_voxel.child_voxels_small
                        [corner_index(0) + voxel_size_ratio * corner_index(1) +
                         voxel_size_ratio * voxel_size_ratio * corner_index(2)];
              }

              const TsdfSubVoxel& const_sub_voxel = sub_voxel;
              if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                        &(corner_sdf(i)))) {
                all_neighbors_observed = false;
                break;
              }

            } else {
              VoxelIndex normal_voxel_offset = VoxelIndex::Zero();

              for (unsigned int j = 0u; j < 3u; j++) {
                if (corner_index(j) >= voxel_size_ratio) {
                  normal_voxel_offset(j) = 1;
                  corner_index(j) = corner_index(j) - voxel_size_ratio;
                }
              }

              VoxelIndex neighbor_normal_voxel_index =
                  normal_voxel_index + normal_voxel_offset;

              if (block->isValidVoxelIndex(neighbor_normal_voxel_index)) {
                const VoxelType& neighbor_normal_voxel =
                    block->getVoxelByVoxelIndex(neighbor_normal_voxel_index);

                TsdfSubVoxel sub_voxel;
                if (extract_from_queried) {
                  CHECK_EQ(
                      neighbor_normal_voxel.child_voxels_queried.size(),
                      voxel_size_ratio * voxel_size_ratio * voxel_size_ratio);
                  sub_voxel = neighbor_normal_voxel.child_voxels_queried
                                  [corner_index(0) +
                                   voxel_size_ratio * corner_index(1) +
                                   voxel_size_ratio * voxel_size_ratio *
                                       corner_index(2)];
                } else {
                  CHECK_EQ(
                      neighbor_normal_voxel.child_voxels_small.size(),
                      voxel_size_ratio * voxel_size_ratio * voxel_size_ratio);
                  sub_voxel = neighbor_normal_voxel.child_voxels_small
                                  [corner_index(0) +
                                   voxel_size_ratio * corner_index(1) +
                                   voxel_size_ratio * voxel_size_ratio *
                                       corner_index(2)];
                }
                const TsdfSubVoxel& const_sub_voxel = sub_voxel;
                if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                          &(corner_sdf(i)))) {
                  all_neighbors_observed = false;
                  break;
                }

              } else {
                // We have to access a different block.
                BlockIndex block_offset = BlockIndex::Zero();

                DCHECK_GE(neighbor_normal_voxel_index.x(), 0);
                DCHECK_GE(neighbor_normal_voxel_index.y(), 0);
                DCHECK_GE(neighbor_normal_voxel_index.z(), 0);

                for (unsigned int j = 0u; j < 3u; j++) {
                  if (neighbor_normal_voxel_index(j) >=
                      static_cast<IndexElement>(voxels_per_side_)) {
                    block_offset(j) = 1;
                    neighbor_normal_voxel_index(j) =
                        neighbor_normal_voxel_index(j) - voxels_per_side_;
                  }
                }

                BlockIndex neighbor_block_index =
                    block->block_index() + block_offset;

                if (sdf_layer_const_->hasBlock(neighbor_block_index)) {
                  const Block<VoxelType>& neighbor_block =
                      sdf_layer_const_->getBlockByIndex(neighbor_block_index);

                  CHECK(neighbor_block.isValidVoxelIndex(
                      neighbor_normal_voxel_index));
                  const VoxelType& neighbor_normal_voxel =
                      neighbor_block.getVoxelByVoxelIndex(
                          neighbor_normal_voxel_index);

                  TsdfSubVoxel sub_voxel;
                  if (extract_from_queried) {
                    CHECK_EQ(
                        neighbor_normal_voxel.child_voxels_queried.size(),
                        voxel_size_ratio * voxel_size_ratio * voxel_size_ratio);
                    sub_voxel = neighbor_normal_voxel.child_voxels_queried
                                    [corner_index(0) +
                                     voxel_size_ratio * corner_index(1) +
                                     voxel_size_ratio * voxel_size_ratio *
                                         corner_index(2)];
                  } else {
                    CHECK_EQ(
                        neighbor_normal_voxel.child_voxels_small.size(),
                        voxel_size_ratio * voxel_size_ratio * voxel_size_ratio);
                    sub_voxel = neighbor_normal_voxel.child_voxels_small
                                    [corner_index(0) +
                                     voxel_size_ratio * corner_index(1) +
                                     voxel_size_ratio * voxel_size_ratio *
                                         corner_index(2)];
                  }

                  const TsdfSubVoxel& const_sub_voxel = sub_voxel;
                  if (!utils::getSdfIfValid(const_sub_voxel, config_.min_weight,
                                            &(corner_sdf(i)))) {
                    all_neighbors_observed = false;
                    break;
                  }
                } else {
                  std::cout << "don't have neighbor block" << std::endl;
                  DCHECK(false);
                  all_neighbors_observed = false;
                  break;
                }
              }
            }
            corner_coords.col(i) =
                coords_current_subvoxel + cube_coord_offsets_subvoxel.col(i);
          }

          if (all_neighbors_observed) {
            MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index,
                                    mesh);
          }
        }
      }
    }
  }

  void updateMeshColor(const Block<VoxelType>& block, Mesh* mesh) {
    DCHECK(mesh != nullptr);

    mesh->colors.clear();
    mesh->colors.resize(mesh->indices.size());

    // Use nearest-neighbor search.
    for (size_t i = 0; i < mesh->vertices.size(); i++) {
      const Point& vertex = mesh->vertices[i];
      VoxelIndex voxel_index = block.computeVoxelIndexFromCoordinates(vertex);
      if (block.isValidVoxelIndex(voxel_index)) {
        const VoxelType& voxel = block.getVoxelByVoxelIndex(voxel_index);
        if (adaptive_mapping) {
          if (voxel.small_type_status.should_be_divided()) {
            Point vertex_coordinate_in_normal_voxel =
                vertex - block.origin() -
                voxel_index.cast<FloatingPoint>() * voxel_size_;

            getColorFromSubVoxels(vertex_coordinate_in_normal_voxel, voxel,
                                  true, mesh, i);

            if (mesh->colors[i].r == 0 && mesh->colors[i].g == 0 &&
                mesh->colors[i].b == 0) {
              utils::getColorIfValid(voxel, config_.min_weight,
                                     &(mesh->colors[i]));
            }
          } else if (voxel.middle_type_status.should_be_divided()) {
            Point vertex_coordinate_in_normal_voxel =
                vertex - block.origin() -
                voxel_index.cast<FloatingPoint>() * voxel_size_;

            getColorFromSubVoxels(vertex_coordinate_in_normal_voxel, voxel,
                                  false, mesh, i);

            if (mesh->colors[i].r == 0 && mesh->colors[i].g == 0 &&
                mesh->colors[i].b == 0) {
              utils::getColorIfValid(voxel, config_.min_weight,
                                     &(mesh->colors[i]));
            }
          } else {
            utils::getColorIfValid(voxel, config_.min_weight,
                                   &(mesh->colors[i]));
          }
        } else {
          utils::getColorIfValid(voxel, config_.min_weight, &(mesh->colors[i]));
        }
      } else {
        const typename Block<VoxelType>::ConstPtr neighbor_block =
            sdf_layer_const_->getBlockPtrByCoordinates(vertex);
        const VoxelType& voxel = neighbor_block->getVoxelByCoordinates(vertex);

        // TODO(jianhao) this is a totally copy of previous one, considering
        // to make it as function
        if (adaptive_mapping) {
          if (voxel.small_type_status.should_be_divided()) {
            Point vertex_coordinate_in_normal_voxel =
                vertex - block.origin() -
                voxel_index.cast<FloatingPoint>() * voxel_size_;

            getColorFromSubVoxels(vertex_coordinate_in_normal_voxel, voxel,
                                  true, mesh, i);

            if (mesh->colors[i].r == 0 && mesh->colors[i].g == 0 &&
                mesh->colors[i].b == 0) {
              utils::getColorIfValid(voxel, config_.min_weight,
                                     &(mesh->colors[i]));
            }
          } else if (voxel.middle_type_status.should_be_divided()) {
            Point vertex_coordinate_in_normal_voxel =
                vertex - block.origin() -
                voxel_index.cast<FloatingPoint>() * voxel_size_;

            getColorFromSubVoxels(vertex_coordinate_in_normal_voxel, voxel,
                                  false, mesh, i);

            if (mesh->colors[i].r == 0 && mesh->colors[i].g == 0 &&
                mesh->colors[i].b == 0) {
              utils::getColorIfValid(voxel, config_.min_weight,
                                     &(mesh->colors[i]));
            }
          } else {
            utils::getColorIfValid(voxel, config_.min_weight,
                                   &(mesh->colors[i]));
          }
        } else {
          utils::getColorIfValid(voxel, config_.min_weight, &(mesh->colors[i]));
        }
      }
    }
  }

  void getColorFromSubVoxels(const Point& vertex_coordinate_in_normal_voxel,
                             const VoxelType& voxel, const bool& from_queried,
                             Mesh* mesh, const size_t& mesh_idx) {
    int sub_voxel_ratio;

    if (from_queried) {
      // we are getting color from the subvoxel of queried semantic voxels
      CHECK(voxel.child_voxels_queried.size() ==
            adaptive_ratio_small * adaptive_ratio_small * adaptive_ratio_small);
      sub_voxel_ratio = adaptive_ratio_small;
    } else {
      // TODO(jianhao) since we only have two levels for now, if it's not
      // queried semantic, then it's small semantic
      CHECK(voxel.child_voxels_small.size() == adaptive_ratio_middle *
                                                   adaptive_ratio_middle *
                                                   adaptive_ratio_middle);
      sub_voxel_ratio = adaptive_ratio_middle;
    }

    FloatingPoint sub_voxel_size_inv =
        static_cast<FloatingPoint>(sub_voxel_ratio) / voxel_size_;

    VoxelIndex sub_voxel_index = VoxelIndex(
        std::floor(vertex_coordinate_in_normal_voxel.x() * sub_voxel_size_inv +
                   1e-6),
        std::floor(vertex_coordinate_in_normal_voxel.y() * sub_voxel_size_inv +
                   1e-6),
        std::floor(vertex_coordinate_in_normal_voxel.z() * sub_voxel_size_inv +
                   1e-6));

    if (from_queried) {
      const TsdfSubVoxel& sub_voxel =
          voxel.child_voxels_queried[sub_voxel_index(0) +
                                     sub_voxel_ratio * sub_voxel_index(1) +
                                     sub_voxel_ratio * sub_voxel_ratio *
                                         sub_voxel_index(2)];
      utils::getColorIfValid(sub_voxel, config_.min_weight,
                             &(mesh->colors[mesh_idx]));
    } else {
      const TsdfSubVoxel& sub_voxel =
          voxel.child_voxels_small[sub_voxel_index(0) +
                                   sub_voxel_ratio * sub_voxel_index(1) +
                                   sub_voxel_ratio * sub_voxel_ratio *
                                       sub_voxel_index(2)];
      utils::getColorIfValid(sub_voxel, config_.min_weight,
                             &(mesh->colors[mesh_idx]));
    }
  }

  void updateMeshColorSemantic(
      const Block<VoxelType>& block, Mesh* mesh,
      const std::shared_ptr<voxblox::ColorMap>& color_map) {
    DCHECK(mesh != nullptr);

    mesh->colors.clear();
    mesh->colors.resize(mesh->indices.size());

    // Use nearest-neighbor search.
    for (size_t i = 0; i < mesh->vertices.size(); i++) {
      const Point& vertex = mesh->vertices[i];
      VoxelIndex voxel_index = block.computeVoxelIndexFromCoordinates(vertex);
      if (block.isValidVoxelIndex(voxel_index)) {
        const VoxelType& voxel = block.getVoxelByVoxelIndex(voxel_index);
        utils::getColorIfValidSemantic(voxel, config_.min_weight,
                                       &(mesh->colors[i]), color_map);
      } else {
        const typename Block<VoxelType>::ConstPtr neighbor_block =
            sdf_layer_const_->getBlockPtrByCoordinates(vertex);
        const VoxelType& voxel = neighbor_block->getVoxelByCoordinates(vertex);
        utils::getColorIfValidSemantic(voxel, config_.min_weight,
                                       &(mesh->colors[i]), color_map);
      }
    }
  }

 protected:
  MeshIntegratorConfig config_;

  /**
   * Having both a const and a mutable pointer to the layer allows this
   * integrator to work both with a const layer (in case you don't want to
   * clear the updated flag) and mutable layer (in case you do want to
   * clear the updated flag).
   */
  Layer<VoxelType>* sdf_layer_mutable_;
  const Layer<VoxelType>* sdf_layer_const_;

  MeshLayer* mesh_layer_;

  // Cached map config.
  FloatingPoint voxel_size_;
  size_t voxels_per_side_;
  FloatingPoint block_size_;

  // Derived types.
  FloatingPoint voxel_size_inv_;
  FloatingPoint voxels_per_side_inv_;
  FloatingPoint block_size_inv_;

  // Cached index map.
  Eigen::Matrix<int, 3, 8> cube_index_offsets_;
  Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets_large_;
  Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets_middle_;
  Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets_small_;

  Eigen::Matrix<int, 2, 12> cube_index_face_;
  Eigen::Matrix<int, 3, 12> axis_index_face_;
  Eigen::Matrix<int, 4, 6> cube_index_edge_;
  Eigen::Matrix<int, 3, 6> axis_index_edge_;
};

}  // namespace voxblox

#endif  // VOXBLOX_MESH_MESH_INTEGRATOR_H_
