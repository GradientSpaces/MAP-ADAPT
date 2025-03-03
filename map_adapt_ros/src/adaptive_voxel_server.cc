#include "map_adapt_ros/adaptive_voxel_server.h"

#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>

#include "map_adapt_ros/conversions.h"
#include "map_adapt_ros/ros_params.h"

namespace voxblox {

VoxelServer::VoxelServer(const ros::NodeHandle& nh,
                         const ros::NodeHandle& nh_private)
    : VoxelServer(nh, nh_private, getTsdfMapConfigFromRosParam(nh_private),
                  getTsdfIntegratorConfigFromRosParam(nh_private),
                  getMeshIntegratorConfigFromRosParam(nh_private)) {}

VoxelServer::VoxelServer(const ros::NodeHandle& nh,
                         const ros::NodeHandle& nh_private,
                         const TsdfMap::Config& config,
                         const TsdfIntegratorBase::Config& integrator_config,
                         const MeshIntegratorConfig& mesh_config)
    : nh_(nh),
      nh_private_(nh_private),
      verbose_(true),
      visualize_semantic(false),
      load_semantic_probability(false),
      load_geo_complexity(false),
      adaptive_mapping(false),
      adaptive_ratio_small(8),
      adaptive_ratio_middle(2),
      semantic_update_method("max_pooling"),
      fuse_semantic_for_subvoxel(false),
      output_dir("/"),
      scene_name("scene1"),
      semantic_level_split_id(0),
      world_frame_("world"),
      icp_corrected_frame_("icp_corrected"),
      pose_corrected_frame_("pose_corrected"),
      max_block_distance_from_body_(std::numeric_limits<FloatingPoint>::max()),
      slice_level_(0.5),
      use_freespace_pointcloud_(false),
      color_map_(new RainbowColorMap()),
      publish_pointclouds_on_update_(false),
      publish_slices_(false),
      publish_pointclouds_(false),
      publish_tsdf_map_(false),
      cache_mesh_(false),
      enable_icp_(false),
      accumulate_icp_corrections_(true),
      pointcloud_queue_size_(1),
      num_subscribers_tsdf_map_(0),
      transformer_(nh, nh_private) {
  getServerConfigFromRosParam(nh_private);

  // Advertise topics.
  surface_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
          "surface_pointcloud", 1, true);
  tsdf_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >("tsdf_pointcloud",
                                                              1, true);
  occupancy_marker_pub_ =
      nh_private_.advertise<visualization_msgs::MarkerArray>("occupied_nodes",
                                                             1, true);
  tsdf_slice_pub_ = nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >(
      "tsdf_slice", 1, true);

  command_to_pc_publisher_pub_ = nh_private_.advertise<std_msgs::String>(
      "command_to_pc_publisher", 5, true);

  nh_private_.param("pointcloud_queue_size", pointcloud_queue_size_,
                    pointcloud_queue_size_);
  pointcloud_sub_ = nh_.subscribe("pointcloud", pointcloud_queue_size_,
                                  &VoxelServer::insertPointcloud, this);

  mesh_pub_ = nh_private_.advertise<voxblox_msgs::Mesh>("mesh", 1, true);
  tsdf_voxel_pub_ = nh_private_.advertise<visualization_msgs::MarkerArray>(
      "tsdf_voxels", 1, true);

  // Publishing/subscribing to a layer from another node (when using this as
  // a library, for example within a planner).
  tsdf_map_pub_ =
      nh_private_.advertise<voxblox_msgs::Layer>("tsdf_map_out", 1, false);
  tsdf_map_sub_ = nh_private_.subscribe("tsdf_map_in", 1,
                                        &VoxelServer::tsdfMapCallback, this);
  nh_private_.param("publish_tsdf_map", publish_tsdf_map_, publish_tsdf_map_);

  if (use_freespace_pointcloud_) {
    // points that are not inside an object, but may also not be on a surface.
    // These will only be used to mark freespace beyond the truncation distance.
    freespace_pointcloud_sub_ =
        nh_.subscribe("freespace_pointcloud", pointcloud_queue_size_,
                      &VoxelServer::insertFreespacePointcloud, this);
  }

  if (enable_icp_) {
    icp_transform_pub_ = nh_private_.advertise<geometry_msgs::TransformStamped>(
        "icp_transform", 1, true);
    nh_private_.param("icp_corrected_frame", icp_corrected_frame_,
                      icp_corrected_frame_);
    nh_private_.param("pose_corrected_frame", pose_corrected_frame_,
                      pose_corrected_frame_);
  }

  // Initialize TSDF Map and integrator.
  tsdf_map_.reset(new TsdfMap(config));

  std::string method("merged");
  nh_private_.param("method", method, method);
  if (method.compare("simple") == 0) {
    tsdf_integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else if (method.compare("merged") == 0) {
    tsdf_integrator_.reset(new MergedTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else if (method.compare("fast") == 0) {
    tsdf_integrator_.reset(new FastTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else {
    tsdf_integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  }

  // Set up for semantics

  std::vector<int> small_semantic;
  nh_private_.getParam("small_semantic", small_semantic);
  tsdf_integrator_->small_semantic.clear();
  if (small_semantic.size() > 0) {
    for (uint id_semantic = 0; id_semantic < small_semantic.size();
         id_semantic++) {
      tsdf_integrator_->small_semantic.push_back(
          (uint32_t)small_semantic[id_semantic]);
    }
  }
  std::vector<int> large_semantic;
  nh_private_.getParam("large_semantic", large_semantic);
  tsdf_integrator_->large_semantic.clear();
  if (large_semantic.size() > 0) {
    for (uint id_semantic = 0; id_semantic < large_semantic.size();
         id_semantic++) {
      tsdf_integrator_->large_semantic.push_back(
          (uint32_t)large_semantic[id_semantic]);
    }
  }

  // Set up for geo complexity
  std::vector<float> geo_thresholds;
  nh_private_.getParam("geo_thresholds", geo_thresholds);
  tsdf_integrator_->geo_thresholds.clear();
  if (geo_thresholds.size() > 0) {
    for (uint id_th = 0; id_th < geo_thresholds.size(); id_th++) {
      tsdf_integrator_->geo_thresholds.push_back(geo_thresholds[id_th]);
    }
  }

  tsdf_integrator_->adaptive_mapping = adaptive_mapping;
  tsdf_integrator_->adaptive_ratio_small = adaptive_ratio_small;
  tsdf_integrator_->adaptive_ratio_middle = adaptive_ratio_middle;
  tsdf_integrator_->semantic_update_method = semantic_update_method;
  tsdf_integrator_->fuse_semantic_for_subvoxel = fuse_semantic_for_subvoxel;
  std::cout << "********Info relating to semantic***************" << std::endl;
  std::cout << "Method to update semantic: "
            << tsdf_integrator_->semantic_update_method << std::endl;
  std::cout << "Whether to adapt the map depending on semantic: "
            << tsdf_integrator_->adaptive_mapping << std::endl;
  std::cout << "Adaptive ratio for queried semantics: " << adaptive_ratio_small
            << std::endl;
  std::cout << "Adaptive ratio for small semantics: " << adaptive_ratio_middle
            << std::endl;
  std::cout << "Whether to fuse semantic on subvoxels: "
            << fuse_semantic_for_subvoxel << std::endl;
  std::cout << "Queried semantic:";
  for (uint id_semantic = 0;
       id_semantic < tsdf_integrator_->small_semantic.size(); id_semantic++) {
    std::cout << " " << tsdf_integrator_->small_semantic[id_semantic];
  }
  std::cout << "Large semantic:";
  for (uint id_semantic = 0;
       id_semantic < tsdf_integrator_->large_semantic.size(); id_semantic++) {
    std::cout << " " << tsdf_integrator_->large_semantic[id_semantic];
  }
  std::cout << "Geo Complexity thresholds:";
  for (uint id_th = 0; id_th < tsdf_integrator_->geo_thresholds.size();
       id_th++) {
    std::cout << " " << tsdf_integrator_->geo_thresholds[id_th];
  }
  std::cout << std::endl;

  mesh_layer_.reset(new MeshLayer(tsdf_map_->block_size()));

  mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(
      mesh_config, tsdf_map_->getTsdfLayerPtr(), mesh_layer_.get()));
  mesh_integrator_->adaptive_mapping = adaptive_mapping;
  mesh_integrator_->adaptive_ratio_small = adaptive_ratio_small;
  mesh_integrator_->adaptive_ratio_middle = adaptive_ratio_middle;
  if (small_semantic.size() > 0) {
    for (uint id_semantic = 0; id_semantic < small_semantic.size();
         id_semantic++) {
      mesh_integrator_->small_semantic.push_back(
          (uint32_t)small_semantic[id_semantic]);
    }
  }
  if (large_semantic.size() > 0) {
    for (uint id_semantic = 0; id_semantic < large_semantic.size();
         id_semantic++) {
      mesh_integrator_->large_semantic.push_back(
          (uint32_t)large_semantic[id_semantic]);
    }
  }

  icp_.reset(new ICP(getICPConfigFromRosParam(nh_private)));

  // Advertise services.
  generate_mesh_srv_ = nh_private_.advertiseService(
      "generate_mesh", &VoxelServer::generateMeshCallback, this);
  generate_rgb_mesh_srv_ = nh_private_.advertiseService(
      "generate_rgb_mesh", &VoxelServer::generateRGBMeshCallback, this);
  generate_semantic_mesh_srv_ = nh_private_.advertiseService(
      "generate_semantic_mesh_srv_", &VoxelServer::generateSemanticMeshCallback,
      this);
  generate_semantic_pc_srv_ = nh_private_.advertiseService(
      "generate_semantic_pc_srv_", &VoxelServer::generateSemanticPcCallback,
      this);
  generate_semantic_probability_srv_ = nh_private_.advertiseService(
      "generate_semantic_probability_srv_",
      &VoxelServer::generateSemanticProbabilityCallback, this);
  display_tsdf_voxels = nh_private_.advertiseService(
      "display_tsdf_voxels", &VoxelServer::displayTSDFVoxelsCallback, this);
  clear_map_srv_ = nh_private_.advertiseService(
      "clear_map", &VoxelServer::clearMapCallback, this);
  save_map_srv_ = nh_private_.advertiseService(
      "save_map", &VoxelServer::saveMapCallback, this);
  load_map_srv_ = nh_private_.advertiseService(
      "load_map", &VoxelServer::loadMapCallback, this);
  publish_pointclouds_srv_ = nh_private_.advertiseService(
      "publish_pointclouds", &VoxelServer::publishPointcloudsCallback, this);
  publish_tsdf_map_srv_ = nh_private_.advertiseService(
      "publish_map", &VoxelServer::publishTsdfMapCallback, this);

  // If set, use a timer to progressively integrate the mesh.
  double update_mesh_every_n_sec = 1.0;
  nh_private_.param("update_mesh_every_n_sec", update_mesh_every_n_sec,
                    update_mesh_every_n_sec);

  if (update_mesh_every_n_sec > 0.0) {
    update_mesh_timer_ =
        nh_private_.createTimer(ros::Duration(update_mesh_every_n_sec),
                                &VoxelServer::updateMeshEvent, this);
  }

  double publish_map_every_n_sec = 1.0;
  nh_private_.param("publish_map_every_n_sec", publish_map_every_n_sec,
                    publish_map_every_n_sec);

  if (publish_map_every_n_sec > 0.0) {
    publish_map_timer_ =
        nh_private_.createTimer(ros::Duration(publish_map_every_n_sec),
                                &VoxelServer::publishMapEvent, this);
  }

  if (verbose_) {
    std::string config = integrator_config.print();
    std::cout << config << std::endl;
  }
}

void VoxelServer::getServerConfigFromRosParam(
    const ros::NodeHandle& nh_private) {
  // Before subscribing, determine minimum time between messages.
  // 0 by default.
  double min_time_between_msgs_sec = 0.0;
  nh_private.param("min_time_between_msgs_sec", min_time_between_msgs_sec,
                   min_time_between_msgs_sec);
  min_time_between_msgs_.fromSec(min_time_between_msgs_sec);

  nh_private.param("max_block_distance_from_body",
                   max_block_distance_from_body_,
                   max_block_distance_from_body_);
  nh_private.param("slice_level", slice_level_, slice_level_);
  nh_private.param("world_frame", world_frame_, world_frame_);
  nh_private.param("publish_pointclouds_on_update",
                   publish_pointclouds_on_update_,
                   publish_pointclouds_on_update_);
  nh_private.param("publish_slices", publish_slices_, publish_slices_);
  nh_private.param("publish_pointclouds", publish_pointclouds_,
                   publish_pointclouds_);

  nh_private.param("use_freespace_pointcloud", use_freespace_pointcloud_,
                   use_freespace_pointcloud_);
  nh_private.param("pointcloud_queue_size", pointcloud_queue_size_,
                   pointcloud_queue_size_);
  nh_private.param("enable_icp", enable_icp_, enable_icp_);
  nh_private.param("accumulate_icp_corrections", accumulate_icp_corrections_,
                   accumulate_icp_corrections_);

  nh_private.param("verbose", verbose_, verbose_);
  nh_private.param("visualize_semantic", visualize_semantic,
                   visualize_semantic);
  nh_private.param("load_semantic_probability", load_semantic_probability,
                   load_semantic_probability);
  nh_private.param("load_geo_complexity", load_geo_complexity,
                   load_geo_complexity);
  nh_private.param("adaptive_mapping", adaptive_mapping, adaptive_mapping);
  nh_private.param("adaptive_ratio_small", adaptive_ratio_small,
                   adaptive_ratio_small);
  DCHECK_GT(adaptive_ratio_small, 1);
  nh_private.param("adaptive_ratio_middle", adaptive_ratio_middle,
                   adaptive_ratio_middle);
  DCHECK_GT(adaptive_ratio_middle, 1);
  nh_private.param("semantic_update_method", semantic_update_method,
                   semantic_update_method);
  nh_private.param("fuse_semantic_for_subvoxel", fuse_semantic_for_subvoxel,
                   fuse_semantic_for_subvoxel);
  nh_private.param("scene_name", scene_name, scene_name);
  nh_private.param("output_dir", output_dir, output_dir);
  nh_private.param("semantic_level_split_id", semantic_level_split_id,
                   semantic_level_split_id);

  // Mesh settings.
  nh_private.param("mesh_filename", mesh_filename_, mesh_filename_);
  std::string color_mode("");
  nh_private.param("color_mode", color_mode, color_mode);
  color_mode_ = getColorModeFromString(color_mode);

  // Color map for intensity pointclouds.
  std::string intensity_colormap("rainbow");
  float intensity_max_value = kDefaultMaxIntensity;
  nh_private.param("intensity_colormap", intensity_colormap,
                   intensity_colormap);
  nh_private.param("intensity_max_value", intensity_max_value,
                   intensity_max_value);

  // Default set in constructor.
  if (intensity_colormap == "rainbow") {
    color_map_.reset(new RainbowColorMap());
  } else if (intensity_colormap == "inverse_rainbow") {
    color_map_.reset(new InverseRainbowColorMap());
  } else if (intensity_colormap == "grayscale") {
    color_map_.reset(new GrayscaleColorMap());
  } else if (intensity_colormap == "inverse_grayscale") {
    color_map_.reset(new InverseGrayscaleColorMap());
  } else if (intensity_colormap == "ironbow") {
    color_map_.reset(new IronbowColorMap());
  } else if (intensity_colormap == "nyu") {
    color_map_.reset(new NYUColorMap());
  } else {
    ROS_ERROR_STREAM("Invalid color map: " << intensity_colormap);
  }
  color_map_->setMaxValue(intensity_max_value);
}

void VoxelServer::processPointCloudMessageAndInsert(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
    const Transformation& T_G_C, const bool is_freespace_pointcloud) {
  // Convert the PCL pointcloud into our awesome format.

  // Horrible hack fix to fix color parsing colors in PCL.
  bool color_pointcloud = false;
  bool has_intensity = false;
  bool has_label = false;
  bool has_probability = false;
  for (size_t d = 0; d < pointcloud_msg->fields.size(); ++d) {
    if (pointcloud_msg->fields[d].name == std::string("rgb")) {
      pointcloud_msg->fields[d].datatype = sensor_msgs::PointField::FLOAT32;
      color_pointcloud = true;
    } else if (pointcloud_msg->fields[d].name == std::string("rgba")) {
      color_pointcloud = true;
    } else if (pointcloud_msg->fields[d].name == std::string("intensity")) {
      has_intensity = true;
    } else if (pointcloud_msg->fields[d].name == std::string("label")) {
      has_label = true;
    } else if (pointcloud_msg->fields[d].name == std::string("top4_label")) {
      has_probability = true;
    }
  }

  Pointcloud points_C;
  Colors colors;
  Semantics semantics;
  std::vector<uint32_t> semantics_encoded;
  std::vector<uint32_t> probabilities_encoded;
  std::vector<float> geo_complexity;
  timing::Timer ptcloud_timer("ptcloud_preprocess");

  // Convert differently depending on RGB or I type.
  if (load_semantic_probability) {
    if (load_geo_complexity) {
      pcl::PointCloud<PointXYZRGBILabelMap> pointcloud_pcl;
      pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
      convertPointcloudGeoSemantic(pointcloud_pcl, color_map_, &points_C,
                                   &colors, &geo_complexity, &semantics_encoded,
                                   &probabilities_encoded);
    } else {
      pcl::PointCloud<PointXYZRGBLabelMap> pointcloud_pcl;
      pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
      convertPointcloudSemanticProbability(
          pointcloud_pcl, color_map_, &points_C, &colors, &semantics_encoded,
          &probabilities_encoded);
    }

  } else if (load_geo_complexity) {
    pcl::PointCloud<PointXYZRGBI> pointcloud_pcl;
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloudGeoComplexity(pointcloud_pcl, color_map_, &points_C,
                                   &colors, &geo_complexity);
  } else if (color_pointcloud) {
    if (has_label) {
      pcl::PointCloud<pcl::PointXYZRGBL> pointcloud_pcl;
      // pointcloud_pcl is modified below:
      pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
      convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
    } else if (has_probability) {
      pcl::PointCloud<PointXYZRGBLabelMap> pointcloud_pcl;
      // pointcloud_pcl is modified below:
      pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
      convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
    } else {
      pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
      // pointcloud_pcl is modified below:
      pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
      convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
    }

  } else if (has_intensity) {
    pcl::PointCloud<pcl::PointXYZI> pointcloud_pcl;
    // pointcloud_pcl is modified below:
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  } else {
    pcl::PointCloud<pcl::PointXYZ> pointcloud_pcl;
    // pointcloud_pcl is modified below:
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  }
  ptcloud_timer.Stop();

  Transformation T_G_C_refined = T_G_C;
  if (enable_icp_) {
    timing::Timer icp_timer("icp");
    if (!accumulate_icp_corrections_) {
      icp_corrected_transform_.setIdentity();
    }
    static Transformation T_offset;
    const size_t num_icp_updates =
        icp_->runICP(tsdf_map_->getTsdfLayer(), points_C,
                     icp_corrected_transform_ * T_G_C, &T_G_C_refined);
    if (verbose_) {
      ROS_INFO("ICP refinement performed %zu successful update steps",
               num_icp_updates);
    }
    icp_corrected_transform_ = T_G_C_refined * T_G_C.inverse();

    if (!icp_->refiningRollPitch()) {
      // its already removed internally but small floating point errors can
      // build up if accumulating transforms
      Transformation::Vector6 T_vec = icp_corrected_transform_.log();
      T_vec[3] = 0.0;
      T_vec[4] = 0.0;
      icp_corrected_transform_ = Transformation::exp(T_vec);
    }

    // Publish transforms as both TF and message.
    tf::Transform icp_tf_msg, pose_tf_msg;
    geometry_msgs::TransformStamped transform_msg;

    tf::transformKindrToTF(icp_corrected_transform_.cast<double>(),
                           &icp_tf_msg);
    tf::transformKindrToTF(T_G_C.cast<double>(), &pose_tf_msg);
    tf::transformKindrToMsg(icp_corrected_transform_.cast<double>(),
                            &transform_msg.transform);
    tf_broadcaster_.sendTransform(
        tf::StampedTransform(icp_tf_msg, pointcloud_msg->header.stamp,
                             world_frame_, icp_corrected_frame_));
    tf_broadcaster_.sendTransform(
        tf::StampedTransform(pose_tf_msg, pointcloud_msg->header.stamp,
                             icp_corrected_frame_, pose_corrected_frame_));

    transform_msg.header.frame_id = world_frame_;
    transform_msg.child_frame_id = icp_corrected_frame_;
    icp_transform_pub_.publish(transform_msg);

    icp_timer.Stop();
  }

  if (verbose_) {
    ROS_INFO("Integrating a pointcloud with %lu points.", points_C.size());
  }

  ros::WallTime start = ros::WallTime::now();
  if (load_semantic_probability) {
    if (load_geo_complexity) {
      ROS_INFO(
          "Process point cloud with geo complexity, semantic information and "
          "probability");
      integratePointcloudGeoSemantic(
          T_G_C_refined, points_C, colors, geo_complexity, semantics_encoded,
          probabilities_encoded, is_freespace_pointcloud);
    } else {
      ROS_INFO("Process point cloud with semantic information and probability");
      integratePointcloudSemanticProbability(
          T_G_C_refined, points_C, colors, semantics_encoded,
          probabilities_encoded, is_freespace_pointcloud);
    }

  } else if (load_geo_complexity) {
    ROS_INFO("Process point cloud with geo complexity information");
    integratePointcloudGeoComplexity(T_G_C_refined, points_C, colors,
                                     geo_complexity, is_freespace_pointcloud);
  } else {
    integratePointcloud(T_G_C_refined, points_C, colors,
                        is_freespace_pointcloud);
  }

  ros::WallTime end = ros::WallTime::now();
  if (verbose_) {
    ROS_INFO("Finished integrating in %f seconds, have %lu blocks.",
             (end - start).toSec(),
             tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks());
  }

  timing::Timer block_remove_timer("remove_distant_blocks");
  tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_block_distance_from_body_);
  mesh_layer_->clearDistantMesh(T_G_C.getPosition(),
                                max_block_distance_from_body_);
  block_remove_timer.Stop();

  // Callback for inheriting classes.
  newPoseCallback(T_G_C);
}

// Checks if we can get the next message from queue.
bool VoxelServer::getNextPointcloudFromQueue(
    std::queue<sensor_msgs::PointCloud2::Ptr>* queue,
    sensor_msgs::PointCloud2::Ptr* pointcloud_msg, Transformation* T_G_C) {
  const size_t kMaxQueueSize = 10;
  if (queue->empty()) {
    return false;
  }
  *pointcloud_msg = queue->front();
  if (transformer_.lookupTransform((*pointcloud_msg)->header.frame_id,
                                   world_frame_,
                                   (*pointcloud_msg)->header.stamp, T_G_C)) {
    queue->pop();
    return true;
  } else {
    if (queue->size() >= kMaxQueueSize) {
      ROS_ERROR_THROTTLE(60,
                         "Input pointcloud queue getting too long! Dropping "
                         "some pointclouds. Either unable to look up transform "
                         "timestamps or the processing is taking too long.");
      while (queue->size() >= kMaxQueueSize) {
        queue->pop();
      }
    }
  }
  return false;
}

void VoxelServer::insertPointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg_in) {
  if (pointcloud_msg_in->header.stamp - last_msg_time_ptcloud_ >
      min_time_between_msgs_) {
    last_msg_time_ptcloud_ = pointcloud_msg_in->header.stamp;
    // So we have to process the queue anyway... Push this back.
    pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::PointCloud2::Ptr pointcloud_msg;
  bool processed_any = false;
  while (
      getNextPointcloudFromQueue(&pointcloud_queue_, &pointcloud_msg, &T_G_C)) {
    constexpr bool is_freespace_pointcloud = false;

    // tell the pc publisher to pause (this is to avoid missing information)
    std_msgs::String msg;
    msg.data = "pause";
    command_to_pc_publisher_pub_.publish(msg);

    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
    processed_any = true;

    // tell the pc publisher to resume publish
    msg.data = "resume";
    command_to_pc_publisher_pub_.publish(msg);
  }

  if (publish_pointclouds_on_update_) {
    publishPointclouds();
  }

  if (verbose_) {
    ROS_INFO_STREAM("Timings: " << std::endl << timing::Timing::Print());
    ROS_INFO_STREAM(
        "Layer memory: " << tsdf_map_->getTsdfLayer().getMemorySize());
  }
}

void VoxelServer::insertFreespacePointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg_in) {
  if (pointcloud_msg_in->header.stamp - last_msg_time_freespace_ptcloud_ >
      min_time_between_msgs_) {
    last_msg_time_freespace_ptcloud_ = pointcloud_msg_in->header.stamp;
    // So we have to process the queue anyway... Push this back.
    freespace_pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::PointCloud2::Ptr pointcloud_msg;
  while (getNextPointcloudFromQueue(&freespace_pointcloud_queue_,
                                    &pointcloud_msg, &T_G_C)) {
    constexpr bool is_freespace_pointcloud = true;
    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
  }
}

void VoxelServer::integratePointcloud(const Transformation& T_G_C,
                                      const Pointcloud& ptcloud_C,
                                      const Colors& colors,
                                      const bool is_freespace_pointcloud) {
  CHECK_EQ(ptcloud_C.size(), colors.size());
  tsdf_integrator_->integratePointCloud(T_G_C, ptcloud_C, colors,
                                        is_freespace_pointcloud);
}

void VoxelServer::integratePointcloudSemanticProbability(
    const Transformation& T_G_C, const Pointcloud& ptcloud_C,
    const Colors& colors, const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool is_freespace_pointcloud) {
  CHECK_EQ(ptcloud_C.size(), colors.size());
  CHECK_EQ(ptcloud_C.size(), semantics_encoded.size());
  CHECK_EQ(ptcloud_C.size(), probabilities_encoded.size());

  tsdf_integrator_->integratePointCloudSemanticProbability(
      T_G_C, ptcloud_C, colors, semantics_encoded, probabilities_encoded,
      is_freespace_pointcloud);

  // updateMesh();

  // displayTSDFVoxels();
}

void VoxelServer::integratePointcloudGeoComplexity(
    const Transformation& T_G_C, const Pointcloud& ptcloud_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const bool is_freespace_pointcloud) {
  CHECK_EQ(ptcloud_C.size(), colors.size());
  CHECK_EQ(ptcloud_C.size(), geo_complexity.size());

  tsdf_integrator_->integratePointcloudGeoComplexity(
      T_G_C, ptcloud_C, colors, geo_complexity, is_freespace_pointcloud);

  // updateMesh();

  // displayTSDFVoxels();
}

void VoxelServer::integratePointcloudGeoSemantic(
    const Transformation& T_G_C, const Pointcloud& ptcloud_C,
    const Colors& colors, const std::vector<float>& geo_complexity,
    const std::vector<uint32_t>& semantics_encoded,
    const std::vector<uint32_t>& probabilities_encoded,
    const bool is_freespace_pointcloud) {
  CHECK_EQ(ptcloud_C.size(), colors.size());
  CHECK_EQ(ptcloud_C.size(), geo_complexity.size());

  tsdf_integrator_->integratePointcloudGeoSemantic(
      T_G_C, ptcloud_C, colors, geo_complexity, semantics_encoded,
      probabilities_encoded, is_freespace_pointcloud);

  // updateMesh();

  // displayTSDFVoxels();
}

void VoxelServer::publishAllUpdatedTsdfVoxels() {
  // Create a pointcloud with distance = intensity.
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  createDistancePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(), &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  tsdf_pointcloud_pub_.publish(pointcloud);
}

void VoxelServer::publishTsdfSurfacePoints() {
  // Create a pointcloud with distance = intensity.
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  const float surface_distance_thresh =
      tsdf_map_->getTsdfLayer().voxel_size() * 0.75;
  createSurfacePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(),
                                       surface_distance_thresh, &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  surface_pointcloud_pub_.publish(pointcloud);
}

void VoxelServer::publishTsdfOccupiedNodes() {
  // Create a pointcloud with distance = intensity.
  visualization_msgs::MarkerArray marker_array;
  createOccupancyBlocksFromTsdfLayer(tsdf_map_->getTsdfLayer(), world_frame_,
                                     &marker_array);
  occupancy_marker_pub_.publish(marker_array);
}

void VoxelServer::publishSlices() {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  createDistancePointcloudFromTsdfLayerSlice(tsdf_map_->getTsdfLayer(), 2,
                                             slice_level_, &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  tsdf_slice_pub_.publish(pointcloud);
}

void VoxelServer::publishMap(bool reset_remote_map) {
  if (!publish_tsdf_map_) {
    return;
  }
  int subscribers = this->tsdf_map_pub_.getNumSubscribers();
  if (subscribers > 0) {
    if (num_subscribers_tsdf_map_ < subscribers) {
      // Always reset the remote map and send all when a new subscriber
      // subscribes. A bit of overhead for other subscribers, but better than
      // inconsistent map states.
      reset_remote_map = true;
    }
    const bool only_updated = !reset_remote_map;
    timing::Timer publish_map_timer("map/publish_tsdf");
    voxblox_msgs::Layer layer_msg;
    serializeLayerAsMsg<TsdfVoxel>(this->tsdf_map_->getTsdfLayer(),
                                   only_updated, &layer_msg);
    if (reset_remote_map) {
      layer_msg.action = static_cast<uint8_t>(MapDerializationAction::kReset);
    }
    this->tsdf_map_pub_.publish(layer_msg);
    publish_map_timer.Stop();
  }
  num_subscribers_tsdf_map_ = subscribers;
}

void VoxelServer::publishPointclouds() {
  // Combined function to publish all possible pointcloud messages -- surface
  // pointclouds, updated points, and occupied points.
  publishAllUpdatedTsdfVoxels();
  publishTsdfSurfacePoints();
  publishTsdfOccupiedNodes();
  if (publish_slices_) {
    publishSlices();
  }
}

void VoxelServer::updateMesh() {
  // if (verbose_) {
  //   ROS_INFO("Updating mesh.");
  // }

  timing::Timer generate_mesh_timer("mesh/update");
  constexpr bool only_mesh_updated_blocks = true;
  constexpr bool clear_updated_flag = true;
  if (visualize_semantic) {
    mesh_integrator_->generateMeshSemantic(only_mesh_updated_blocks,
                                           clear_updated_flag, color_map_);
  } else {
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  }
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");

  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  if (cache_mesh_) {
    cached_mesh_msg_ = mesh_msg;
  }

  publish_mesh_timer.Stop();

  if (publish_pointclouds_ && !publish_pointclouds_on_update_) {
    publishPointclouds();
  }
}

bool VoxelServer::generateMesh() {
  timing::Timer generate_mesh_timer("mesh/generate");
  const bool clear_mesh = true;
  if (clear_mesh) {
    constexpr bool only_mesh_updated_blocks = false;
    constexpr bool clear_updated_flag = true;
    if (visualize_semantic) {
      mesh_integrator_->generateMeshSemantic(only_mesh_updated_blocks,
                                             clear_updated_flag, color_map_);
    } else {
      if (adaptive_mapping) {
        mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                       clear_updated_flag);
      } else {
        mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                       clear_updated_flag);
      }
    }

  } else {
    constexpr bool only_mesh_updated_blocks = true;
    constexpr bool clear_updated_flag = true;
    if (visualize_semantic) {
      mesh_integrator_->generateMeshSemantic(only_mesh_updated_blocks,
                                             clear_updated_flag, color_map_);
    } else {
      if (adaptive_mapping) {
        mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                       clear_updated_flag);
      } else {
        mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                       clear_updated_flag);
      }
    }
  }
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");
  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  publish_mesh_timer.Stop();

  if (!mesh_filename_.empty()) {
    timing::Timer output_mesh_timer("mesh/output");
    const bool success = outputMeshLayerAsPly(mesh_filename_, *mesh_layer_);
    output_mesh_timer.Stop();
    if (success) {
      ROS_INFO("Output file as PLY: %s", mesh_filename_.c_str());
    } else {
      ROS_INFO("Failed to output mesh as PLY: %s", mesh_filename_.c_str());
    }
  }

  ROS_INFO_STREAM("Mesh Timings: " << std::endl << timing::Timing::Print());
  return true;
}

bool VoxelServer::generateMeshSplit(const std::string& file_path) {
  timing::Timer generate_mesh_timer("mesh/generate_split");

  constexpr bool only_mesh_updated_blocks = false;
  constexpr bool clear_updated_flag = true;
  for (int save_type_id = 1; save_type_id < 4; ++save_type_id) {
    timing::Timer t1("mesh/t1");
    mesh_integrator_->generateMeshSelective(only_mesh_updated_blocks,
                                            clear_updated_flag, save_type_id);
    t1.Stop();
    timing::Timer t2("mesh/t2");
    const bool success = outputMeshLayerAsPly(
        file_path + "_level_" + std::to_string(save_type_id) + ".ply",
        *mesh_layer_);
    if (success) {
      ROS_INFO("Output file as PLY: %s", mesh_filename_.c_str());
    } else {
      ROS_INFO("Failed to output mesh as PLY: %s",
               file_path + "_level_" + std::to_string(save_type_id) + ".ply");
    }
    t2.Stop();
  }

  generate_mesh_timer.Stop();

  return true;
}

bool VoxelServer::saveMap(const std::string& file_path) {
  // Inheriting classes should add saving other layers to this function.
  return io::SaveLayer(tsdf_map_->getTsdfLayer(), file_path);
}

bool VoxelServer::loadMap(const std::string& file_path) {
  // Inheriting classes should add other layers to load, as this will only
  // load
  // the TSDF layer.
  constexpr bool kMulitpleLayerSupport = true;
  bool success = io::LoadBlocksFromFile(
      file_path, Layer<TsdfVoxel>::BlockMergingStrategy::kReplace,
      kMulitpleLayerSupport, tsdf_map_->getTsdfLayerPtr());
  if (success) {
    LOG(INFO) << "Successfully loaded TSDF layer.";
  }
  return success;
}

bool VoxelServer::clearMapCallback(std_srvs::Empty::Request& /*request*/,
                                   std_srvs::Empty::Response&
                                   /*response*/) {  // NOLINT
  clear();
  return true;
}

bool VoxelServer::generateMeshCallback(std_srvs::Empty::Request& /*request*/,
                                       std_srvs::Empty::Response&
                                       /*response*/) {  // NOLINT
  return generateMesh();
}

bool VoxelServer::generateRGBMeshCallback(std_srvs::Empty::Request& /*request*/,
                                          std_srvs::Empty::Response&
                                          /*response*/) {  // NOLINT
  timing::Timer generate_mesh_timer("mesh/generate");

  constexpr bool only_mesh_updated_blocks = false;
  constexpr bool clear_updated_flag = true;

  if (adaptive_mapping) {
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  } else {
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  }

  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");
  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  publish_mesh_timer.Stop();

  ROS_INFO_STREAM("Mesh Timings: " << std::endl << timing::Timing::Print());
  return true;
}

bool VoxelServer::generateSemanticMeshCallback(
    std_srvs::Empty::Request& /*request*/, std_srvs::Empty::Response&
    /*response*/) {  // NOLINT
  timing::Timer generate_mesh_timer("mesh/generate");

  constexpr bool only_mesh_updated_blocks = false;
  constexpr bool clear_updated_flag = true;

  mesh_integrator_->generateMeshSemantic(only_mesh_updated_blocks,
                                         clear_updated_flag, color_map_);

  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");
  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  publish_mesh_timer.Stop();

  ROS_INFO_STREAM("Mesh Timings: " << std::endl << timing::Timing::Print());
  return true;
}

bool VoxelServer::generateSemanticPcCallback(
    std_srvs::Empty::Request& /*request*/, std_srvs::Empty::Response&
    /*response*/) {  // NOLINT
  return true;
}

bool VoxelServer::generateSemanticProbabilityCallback(
    std_srvs::Empty::Request& /*request*/, std_srvs::Empty::Response&
    /*response*/) {  // NOLINT

  std::ofstream OutFile(
      "/local/home/zheng/catkin_ws/src/voxblox/map_adapt_ros/mesh_results/"
      "debug_semantic_probability.txt",
      std::ios::out);

  BlockIndexList all_tsdf_blocks;
  tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&all_tsdf_blocks);
  for (size_t idx = 0; idx < all_tsdf_blocks.size(); idx++) {
    const BlockIndex& block_idx = all_tsdf_blocks[idx];
    Block<TsdfVoxel>::Ptr block =
        tsdf_map_->getTsdfLayerPtr()->getBlockPtrByIndex(block_idx);
    if (!block) {
      std::cout << "Trying to query a non-existent block at index: "
                << block_idx.transpose();
      return false;
    }
    IndexElement vps = block->voxels_per_side();
    VoxelIndex voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < vps; ++voxel_index.x()) {
      for (voxel_index.y() = 0; voxel_index.y() < vps; ++voxel_index.y()) {
        for (voxel_index.z() = 0; voxel_index.z() < vps; ++voxel_index.z()) {
          TsdfVoxel* voxel = &(block->getVoxelByVoxelIndex(voxel_index));
          if (voxel->labels.size() == 0) continue;
          CHECK_EQ(voxel->labels.size(), voxel->probabilities.size());

          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
          float x, y, z;
          x = coords(0);
          y = coords(1);
          z = coords(2);

          float n_x, n_y, n_z;
          if (voxel->gradient.size() == 0) {
            n_x = 0;
            n_y = 0;
            n_z = 0;
          } else {
            CHECK_EQ(voxel->gradient.size(), 3);
            n_x = voxel->gradient[0];
            n_y = voxel->gradient[1];
            n_z = voxel->gradient[2];
          }
          OutFile << "start a voxel" << std::endl;
          OutFile << x << " " << y << " " << z << "; ";
          OutFile << static_cast<uint32_t>(voxel->color.r) << " "
                  << static_cast<uint32_t>(voxel->color.g) << " "
                  << static_cast<uint32_t>(voxel->color.b) << " "
                  << static_cast<uint32_t>(voxel->color.a) << "; ";
          OutFile << n_x << " " << n_y << " " << n_z << "; ";
          for (int semantic_id = 0; semantic_id < voxel->labels.size();
               ++semantic_id) {
            OutFile << voxel->labels[semantic_id] << " "
                    << voxel->probabilities[semantic_id] << "; ";
          }
          OutFile << "0 " << voxel->rest_probabilities << " " << std::endl;
        }
      }
    }
  }
  return true;
}

void VoxelServer::outputVoxelsToTxt(const std::string& file_path) {
  std::ofstream OutFile1(file_path + "normal_voxel.txt");
  std::ofstream OutFile2(file_path + "sub_voxel.txt");
  std::ofstream OutFile3(file_path + "semantic_voxel.txt");
  BlockIndexList all_tsdf_blocks;
  tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&all_tsdf_blocks);
  for (size_t idx = 0; idx < all_tsdf_blocks.size(); idx++) {
    const BlockIndex& block_idx = all_tsdf_blocks[idx];
    Block<TsdfVoxel>::Ptr block =
        tsdf_map_->getTsdfLayerPtr()->getBlockPtrByIndex(block_idx);
    if (!block) {
      std::cout << "Trying to query a non-existent block at index: "
                << block_idx.transpose();
      return;
    }
    IndexElement vps = block->voxels_per_side();
    VoxelIndex voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < vps; ++voxel_index.x()) {
      for (voxel_index.y() = 0; voxel_index.y() < vps; ++voxel_index.y()) {
        for (voxel_index.z() = 0; voxel_index.z() < vps; ++voxel_index.z()) {
          TsdfVoxel* voxel = &(block->getVoxelByVoxelIndex(voxel_index));

          if ((voxel->weight <= 1e-4f) ||
              (std::abs(voxel->distance) > 2 * block->voxel_size())) {
            // avoid wasting disks for those far away from the surfaces or
            // have no weight information
            continue;
          }

          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
          float x, y, z;
          x = coords(0);
          y = coords(1);
          z = coords(2);

          OutFile1 << x << " " << y << " " << z << " " << voxel->distance << " "
                   << voxel->weight << " ";
          OutFile1 << static_cast<uint32_t>(voxel->color.r) << " "
                   << static_cast<uint32_t>(voxel->color.g) << " "
                   << static_cast<uint32_t>(voxel->color.b) << " "
                   << static_cast<uint32_t>(voxel->color.a);

          OutFile1 << " " << voxel->geo_complexity;

          for (int semantic_id = 0; semantic_id < voxel->labels.size();
               ++semantic_id) {
            OutFile1 << " " << voxel->labels[semantic_id] << ":"
                     << voxel->probabilities[semantic_id];
          }
          OutFile1 << " 0:" << voxel->rest_probabilities << std::endl;

          if (adaptive_mapping) {
            OutFile2 << x << " " << y << " " << z << std::endl;
            if (voxel->small_type_status.should_be_divided()) {
              OutFile2 << "Has queried subvoxels (type:"
                       << voxel->small_type_status.is_this_type << ",neighbor:"
                       << voxel->small_type_status.is_neighbor_of_this_type()
                       << "): ";
              for (int sub_voxel_idx = 0;
                   sub_voxel_idx < voxel->child_voxels_queried.size();
                   ++sub_voxel_idx) {
                OutFile2 << voxel->child_voxels_queried[sub_voxel_idx].distance
                         << " "
                         << voxel->child_voxels_queried[sub_voxel_idx].weight
                         << ";";
              }
              OutFile2 << std::endl;
            }

            if (voxel->middle_type_status.should_be_divided()) {
              OutFile2 << "Has middle subvoxels (type:"
                       << voxel->middle_type_status.is_this_type << ",neighbor:"
                       << voxel->middle_type_status.is_neighbor_of_this_type()
                       << "): ";
              for (int sub_voxel_idx = 0;
                   sub_voxel_idx < voxel->child_voxels_small.size();
                   ++sub_voxel_idx) {
                OutFile2 << voxel->child_voxels_small[sub_voxel_idx].distance
                         << " "
                         << voxel->child_voxels_small[sub_voxel_idx].weight
                         << ";";
              }
              OutFile2 << std::endl;
            }
          }

          OutFile3 << x << " " << y << " " << z;
          if (voxel->labels.size() > 0) {
            uint32_t this_label;
            SemanticProbabilities::iterator most_likely = std::max_element(
                voxel->probabilities.begin(), voxel->probabilities.end());
            this_label = voxel->labels[std::distance(
                voxel->probabilities.begin(), most_likely)];
            OutFile3 << " " << this_label << std::endl;
          } else {
            OutFile3 << " 0" << std::endl;
          }

          if (voxel->small_type_status.should_be_divided()) {
            OutFile3 << "Has queried subvoxels: ";
            for (int sub_voxel_idx = 0;
                 sub_voxel_idx < voxel->child_voxels_queried.size();
                 ++sub_voxel_idx) {
              TsdfSubVoxel child_voxel =
                  voxel->child_voxels_queried[sub_voxel_idx];
              if ((child_voxel.weight > 1e-4f) &&
                  (std::abs(child_voxel.distance) <=
                   std::sqrt(3) * 0.5 * block->voxel_size() /
                       adaptive_ratio_small) &&
                  (child_voxel.labels.size() > 0)) {
                uint32_t this_label;
                SemanticProbabilities::iterator most_likely =
                    std::max_element(child_voxel.probabilities.begin(),
                                     child_voxel.probabilities.end());
                this_label = child_voxel.labels[std::distance(
                    child_voxel.probabilities.begin(), most_likely)];
                OutFile3 << " " << this_label;
              } else {
                OutFile3 << " 0";
              }
            }
            OutFile3 << std::endl;
          }

          if (voxel->middle_type_status.should_be_divided()) {
            OutFile3 << "Has middle subvoxels: ";
            for (int sub_voxel_idx = 0;
                 sub_voxel_idx < voxel->child_voxels_small.size();
                 ++sub_voxel_idx) {
              TsdfSubVoxel child_voxel =
                  voxel->child_voxels_small[sub_voxel_idx];
              if ((child_voxel.weight > 1e-4f) &&
                  (std::abs(child_voxel.distance) <=
                   std::sqrt(3) * 0.5 * block->voxel_size() /
                       adaptive_ratio_middle) &&
                  (child_voxel.labels.size() > 0)) {
                uint32_t this_label;
                SemanticProbabilities::iterator most_likely =
                    std::max_element(child_voxel.probabilities.begin(),
                                     child_voxel.probabilities.end());
                this_label = child_voxel.labels[std::distance(
                    child_voxel.probabilities.begin(), most_likely)];
                OutFile3 << " " << this_label;
              } else {
                OutFile3 << " 0";
              }
            }
            OutFile3 << std::endl;
          }
        }
      }
    }
  }

  OutFile1.close();
  OutFile2.close();
  OutFile3.close();
}

void VoxelServer::outputMemoryRunningTimeToTxt(const std::string& file_path) {
  std::ofstream OutFile(file_path + "time_memory.txt");

  // Calculate memory size
  size_t size = 0u;

  // Calculate size of members
  size += sizeof(tsdf_map_->getTsdfLayer().voxel_size());
  size += sizeof(tsdf_map_->getTsdfLayer().voxels_per_side());
  size += sizeof(tsdf_map_->getTsdfLayer().block_size());
  size += sizeof(tsdf_map_->getTsdfLayer().block_size_inv());

  BlockIndexList all_tsdf_blocks;

  tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&all_tsdf_blocks);

  for (size_t idx = 0; idx < all_tsdf_blocks.size(); idx++) {
    const BlockIndex& block_idx = all_tsdf_blocks[idx];
    Block<TsdfVoxel>::ConstPtr block =
        tsdf_map_->getTsdfLayer().getBlockPtrByIndex(block_idx);

    size += sizeof(block->voxels_per_side());
    size += sizeof(block->voxel_size());
    size += sizeof(block->origin());
    size += sizeof(block->num_voxels());
    size += sizeof(block->voxel_size_inv());
    size += sizeof(block->block_size());
    size += sizeof(block->has_data());
    size += sizeof(block->updated());

    IndexElement vps = block->voxels_per_side();

    VoxelIndex voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < vps; ++voxel_index.x()) {
      for (voxel_index.y() = 0; voxel_index.y() < vps; ++voxel_index.y()) {
        for (voxel_index.z() = 0; voxel_index.z() < vps; ++voxel_index.z()) {
          TsdfVoxel voxel = block->getVoxelByVoxelIndex(voxel_index);

          size += sizeof(voxel.distance);
          size += sizeof(voxel.weight);
          size += sizeof(voxel.color);
          if (voxel.labels.size() > 0) {
            CHECK_EQ(voxel.labels.size(), voxel.probabilities.size());

            size += (voxel.labels.capacity() * sizeof(voxel.labels[0]));
            size += (voxel.labels.capacity() * sizeof(voxel.probabilities[0]));
          }

          size += sizeof(voxel.rest_probabilities);

          if (adaptive_mapping) {
            // size += sizeof(voxel.small_type_status);
            // size += sizeof(voxel.middle_type_status);

            // voxel.small_type_status can be replaced by two bool+4 uint8, just
            // haven't implemented
            size += 6u;
            size += 6u;
            if (voxel.child_voxels_small.size() > 0) {
              for (int sub_voxel_idx = 0;
                   sub_voxel_idx < voxel.child_voxels_small.size();
                   ++sub_voxel_idx) {
                TsdfSubVoxel subvoxel = voxel.child_voxels_small[sub_voxel_idx];
                size += sizeof(subvoxel.distance);
                size += sizeof(subvoxel.weight);
                size += sizeof(subvoxel.color);
                if (subvoxel.labels.size() > 0) {
                  CHECK_EQ(subvoxel.labels.size(),
                           subvoxel.probabilities.size());

                  size +=
                      (subvoxel.labels.capacity() * sizeof(subvoxel.labels[0]));
                  size += (subvoxel.labels.capacity() *
                           sizeof(subvoxel.probabilities[0]));
                }

                size += sizeof(subvoxel.rest_probabilities);
              }
            }

            if (voxel.child_voxels_queried.size() > 0) {
              for (int sub_voxel_idx = 0;
                   sub_voxel_idx < voxel.child_voxels_queried.size();
                   ++sub_voxel_idx) {
                TsdfSubVoxel subvoxel =
                    voxel.child_voxels_queried[sub_voxel_idx];
                size += sizeof(subvoxel.distance);
                size += sizeof(subvoxel.weight);
                size += sizeof(subvoxel.color);
                if (subvoxel.labels.size() > 0) {
                  CHECK_EQ(subvoxel.labels.size(),
                           subvoxel.probabilities.size());

                  size +=
                      (subvoxel.labels.capacity() * sizeof(subvoxel.labels[0]));
                  size += (subvoxel.labels.capacity() *
                           sizeof(subvoxel.probabilities[0]));
                }

                size += sizeof(subvoxel.rest_probabilities);
              }
            }
          }
        }
      }
    }
  }

  ROS_INFO("Total memory size of the map is: %lu Bytes.", size);
  OutFile << "Total memory size of the map is: " << size << " Bytes."
          << std::endl;

  OutFile << timing::Timing::Print() << std::endl;

  OutFile.close();
}

bool VoxelServer::displayTSDFVoxelsCallback(
    std_srvs::Empty::Request& /*request*/, std_srvs::Empty::Response&
    /*response*/) {  // NOLINT

  displayTSDFVoxels();

  return true;
}

void VoxelServer::displayTSDFVoxels() {
  visualization_msgs::MarkerArray marker_array;
  int marker_idx = 0;

  const Layer<TsdfVoxel>& tsdf_layer = tsdf_map_->getTsdfLayer();
  size_t vps = tsdf_layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;
  FloatingPoint normal_voxel_size = tsdf_layer.voxel_size();

  BlockIndexList all_tsdf_blocks;
  tsdf_layer.getAllAllocatedBlocks(&all_tsdf_blocks);

  for (const BlockIndex& index : all_tsdf_blocks) {
    // Iterate over all voxels in said blocks.
    const Block<TsdfVoxel>& block = tsdf_layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      const TsdfVoxel normal_voxel = block.getVoxelByLinearIndex(linear_index);

      uint32_t most_likely_class = 0;

      if (normal_voxel.labels.size() > 0) {
        SemanticProbabilities::const_iterator most_likely =
            std::max_element(normal_voxel.probabilities.begin(),
                             normal_voxel.probabilities.end());
        most_likely_class = normal_voxel.labels[std::distance(
            normal_voxel.probabilities.begin(), most_likely)];
      } else {
        continue;
      }

      if (std::count(tsdf_integrator_->small_semantic.begin(),
                     tsdf_integrator_->small_semantic.end(),
                     most_likely_class)) {
        if (normal_voxel.weight > 1e-4 &&
            std::abs(normal_voxel.distance) <=
                normal_voxel_size * std::sqrt(3)) {
          visualization_msgs::Marker marker;
          marker.header.frame_id = "world";
          marker.header.stamp = ros::Time::now();
          marker.id = marker_idx;
          marker_idx++;
          marker.type = visualization_msgs::Marker::CUBE;
          marker.action = visualization_msgs::Marker::ADD;

          marker.pose.position.x = coord.x();
          marker.pose.position.y = coord.y();
          marker.pose.position.z = coord.z();

          marker.pose.orientation.x = 0.0;
          marker.pose.orientation.y = 0.0;
          marker.pose.orientation.z = 0.0;
          marker.pose.orientation.w = 1.0;

          marker.scale.x = normal_voxel_size;
          marker.scale.y = normal_voxel_size;
          marker.scale.z = normal_voxel_size;

          marker.color.a = 0.3;
          marker.color.r = 168.0 / 255.0;
          marker.color.g = 30.0 / 255.0;
          marker.color.b = 50.0 / 255.0;

          marker_array.markers.push_back(marker);
        }

      } else if (!std::count(tsdf_integrator_->large_semantic.begin(),
                             tsdf_integrator_->large_semantic.end(),
                             most_likely_class) &&
                 normal_voxel.labels.size() > 0) {
        if (normal_voxel.weight > 1e-4 &&
            std::abs(normal_voxel.distance) <=
                normal_voxel_size * std::sqrt(3)) {
          visualization_msgs::Marker marker;
          marker.header.frame_id = "world";
          marker.header.stamp = ros::Time::now();
          marker.id = marker_idx;
          marker_idx++;
          marker.type = visualization_msgs::Marker::CUBE;
          marker.action = visualization_msgs::Marker::ADD;

          marker.pose.position.x = coord.x();
          marker.pose.position.y = coord.y();
          marker.pose.position.z = coord.z();

          marker.pose.orientation.x = 0.0;
          marker.pose.orientation.y = 0.0;
          marker.pose.orientation.z = 0.0;
          marker.pose.orientation.w = 1.0;

          marker.scale.x = normal_voxel_size;
          marker.scale.y = normal_voxel_size;
          marker.scale.z = normal_voxel_size;

          marker.color.a = 0.3;
          marker.color.r = 23.0 / 255.0;
          marker.color.g = 128.0 / 255.0;
          marker.color.b = 105.0 / 255.0;

          marker_array.markers.push_back(marker);
        }
      } else if (normal_voxel.weight > 1e-4 &&
                 std::abs(normal_voxel.distance) <=
                     normal_voxel_size * std::sqrt(3)) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.id = marker_idx;
        marker_idx++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = coord.x();
        marker.pose.position.y = coord.y();
        marker.pose.position.z = coord.z();

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = normal_voxel_size;
        marker.scale.y = normal_voxel_size;
        marker.scale.z = normal_voxel_size;

        marker.color.a = 0.3;
        marker.color.r = 26.0 / 255.0;
        marker.color.g = 85.0 / 255.0;
        marker.color.b = 153.0 / 255.0;

        marker_array.markers.push_back(marker);
      }
    }
  }

  tsdf_voxel_pub_.publish(marker_array);
}

bool VoxelServer::saveMapCallback(voxblox_msgs::FilePath::Request& request,
                                  voxblox_msgs::FilePath::Response&
                                  /*response*/) {  // NOLINT
  return saveMap(request.file_path);
}

bool VoxelServer::loadMapCallback(voxblox_msgs::FilePath::Request& request,
                                  voxblox_msgs::FilePath::Response&
                                  /*response*/) {  // NOLINT
  bool success = loadMap(request.file_path);
  return success;
}

bool VoxelServer::publishPointcloudsCallback(
    std_srvs::Empty::Request& /*request*/, std_srvs::Empty::Response&
    /*response*/) {  // NOLINT
  publishPointclouds();
  return true;
}

bool VoxelServer::publishTsdfMapCallback(std_srvs::Empty::Request& /*request*/,
                                         std_srvs::Empty::Response&
                                         /*response*/) {  // NOLINT
  publishMap();
  return true;
}

void VoxelServer::updateMeshEvent(const ros::TimerEvent& /*event*/) {
  updateMesh();
}

void VoxelServer::publishMapEvent(const ros::TimerEvent& /*event*/) {
  publishMap();
}

void VoxelServer::clear() {
  tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
  mesh_layer_->clear();

  // Publish a message to reset the map to all subscribers.
  if (publish_tsdf_map_) {
    constexpr bool kResetRemoteMap = true;
    publishMap(kResetRemoteMap);
  }
}

void VoxelServer::tsdfMapCallback(const voxblox_msgs::Layer& layer_msg) {
  timing::Timer receive_map_timer("map/receive_tsdf");

  bool success =
      deserializeMsgToLayer<TsdfVoxel>(layer_msg, tsdf_map_->getTsdfLayerPtr());

  if (!success) {
    ROS_ERROR_THROTTLE(10, "Got an invalid TSDF map message!");
  } else {
    ROS_INFO_ONCE("Got an TSDF map from ROS topic!");
    if (publish_pointclouds_on_update_) {
      publishPointclouds();
    }
  }
}

void VoxelServer::updateOutputFileNames() {
  std::string part1 =
      adaptive_mapping
          ? "Adaptive_split_" + std::to_string(semantic_level_split_id)
          : "Fix";
  if (load_geo_complexity) {
    part1 = part1 + "_geo_complexity";
  }
  std::string part2 = std::to_string(
      (int)std::round(tsdf_map_->getTsdfLayer().voxel_size() * 100.0));
  std::string part4 = "est_pose_est_semantic";

  mesh_filename_ = output_dir + "/" + scene_name + "/" + part1 + "_" + part2 +
                   "cm_" + "_" + part4 + ".ply";
}

void VoxelServer::saveeverything() {
  std::string part1 =
      adaptive_mapping
          ? "Adaptive_split_" + std::to_string(semantic_level_split_id)
          : "Fix";
  if (load_geo_complexity) {
    part1 = part1 + "_geo_complexity";
  }
  std::string part2 = std::to_string(
      (int)std::round(tsdf_map_->getTsdfLayer().voxel_size() * 100.0));
  std::string part4 = "est_pose_est_semantic";

  outputVoxelsToTxt(output_dir + "/" + scene_name + "/" + part1 + "_" + part2 +
                    "cm_" + "_" + part4 + "_");

  outputMemoryRunningTimeToTxt(output_dir + "/" + scene_name + "/" + part1 +
                               "_" + part2 + "cm_" + "_" + part4 + "_");
}

}  // namespace voxblox
