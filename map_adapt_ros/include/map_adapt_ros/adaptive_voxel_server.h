#ifndef ADAPTIVE_VOXEL_SERVER_H_
#define ADAPTIVE_VOXEL_SERVER_H_

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <voxblox/alignment/icp.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox_msgs/FilePath.h>
#include <voxblox_msgs/Mesh.h>

#include <memory>
#include <queue>
#include <string>

#include "map_adapt_ros/mesh_vis.h"
#include "map_adapt_ros/ptcloud_vis.h"
#include "map_adapt_ros/transformer.h"

namespace voxblox {

constexpr float kDefaultMaxIntensity = 100.0;

class VoxelServer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VoxelServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  VoxelServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
              const TsdfMap::Config& config,
              const TsdfIntegratorBase::Config& integrator_config,
              const MeshIntegratorConfig& mesh_config);
  virtual ~VoxelServer() {}

  void getServerConfigFromRosParam(const ros::NodeHandle& nh_private);

  void insertPointcloud(const sensor_msgs::PointCloud2::Ptr& pointcloud);

  void insertFreespacePointcloud(
      const sensor_msgs::PointCloud2::Ptr& pointcloud);

  virtual void processPointCloudMessageAndInsert(
      const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
      const Transformation& T_G_C, const bool is_freespace_pointcloud);

  void integratePointcloud(const Transformation& T_G_C,
                           const Pointcloud& ptcloud_C, const Colors& colors,
                           const bool is_freespace_pointcloud = false);

  void integratePointcloudSemanticProbability(
      const Transformation& T_G_C, const Pointcloud& ptcloud_C,
      const Colors& colors, const std::vector<uint32_t>& semantics_encoded,
      const std::vector<uint32_t>& probabilities_encoded,
      const bool is_freespace_pointcloud = false);

  void integratePointcloudGeoComplexity(
      const Transformation& T_G_C, const Pointcloud& ptcloud_C,
      const Colors& colors, const std::vector<float>& geo_complexity,
      const bool is_freespace_pointcloud = false);

  void integratePointcloudGeoSemantic(
      const Transformation& T_G_C, const Pointcloud& ptcloud_C,
      const Colors& colors, const std::vector<float>& geo_complexity,
      const std::vector<uint32_t>& semantics_encoded,
      const std::vector<uint32_t>& probabilities_encoded,
      const bool is_freespace_pointcloud = false);

  virtual void newPoseCallback(const Transformation& /*new_pose*/) {
    // Do nothing.
  }

  void publishAllUpdatedTsdfVoxels();
  void publishTsdfSurfacePoints();
  void publishTsdfOccupiedNodes();

  virtual void publishSlices();
  /// Incremental update.
  virtual void updateMesh();
  /// Batch update.
  virtual bool generateMesh();
  virtual bool generateMeshSplit(const std::string& file_path);
  // Publishes all available pointclouds.
  virtual void publishPointclouds();
  // Publishes the complete map
  virtual void publishMap(bool reset_remote_map = false);
  virtual bool saveMap(const std::string& file_path);
  virtual bool loadMap(const std::string& file_path);

  bool clearMapCallback(std_srvs::Empty::Request& request,           // NOLINT
                        std_srvs::Empty::Response& response);        // NOLINT
  bool saveMapCallback(voxblox_msgs::FilePath::Request& request,     // NOLINT
                       voxblox_msgs::FilePath::Response& response);  // NOLINT
  bool loadMapCallback(voxblox_msgs::FilePath::Request& request,     // NOLINT
                       voxblox_msgs::FilePath::Response& response);  // NOLINT
  bool generateMeshCallback(std_srvs::Empty::Request& request,       // NOLINT
                            std_srvs::Empty::Response& response);    // NOLINT
  bool generateRGBMeshCallback(std_srvs::Empty::Request& request,
                               std_srvs::Empty::Response& response);  // NOLINT
  bool generateSemanticMeshCallback(
      std_srvs::Empty::Request& request,
      std_srvs::Empty::Response& response);  // NOLINT
  bool generateSemanticPcCallback(
      std_srvs::Empty::Request& request,
      std_srvs::Empty::Response& response);  // NOLINT
  bool generateSemanticProbabilityCallback(
      std_srvs::Empty::Request& request,
      std_srvs::Empty::Response& response);  // NOLINT
  void outputVoxelsToTxt(const std::string& file_path);
  void outputMemoryRunningTimeToTxt(const std::string& file_path);
  bool displayTSDFVoxelsCallback(
      std_srvs::Empty::Request& request,
      std_srvs::Empty::Response& response);  // NOLINT
  bool publishPointcloudsCallback(
      std_srvs::Empty::Request& request,                             // NOLINT
      std_srvs::Empty::Response& response);                          // NOLINT
  bool publishTsdfMapCallback(std_srvs::Empty::Request& request,     // NOLINT
                              std_srvs::Empty::Response& response);  // NOLINT

  void displayTSDFVoxels();

  void updateMeshEvent(const ros::TimerEvent& event);
  void publishMapEvent(const ros::TimerEvent& event);

  std::shared_ptr<TsdfMap> getTsdfMapPtr() { return tsdf_map_; }
  std::shared_ptr<const TsdfMap> getTsdfMapPtr() const { return tsdf_map_; }

  /// Accessors for setting and getting parameters.
  double getSliceLevel() const { return slice_level_; }
  void setSliceLevel(double slice_level) { slice_level_ = slice_level; }

  bool setPublishSlices() const { return publish_slices_; }
  void setPublishSlices(const bool publish_slices) {
    publish_slices_ = publish_slices;
  }

  void setWorldFrame(const std::string& world_frame) {
    world_frame_ = world_frame;
  }
  std::string getWorldFrame() const { return world_frame_; }

  /// CLEARS THE ENTIRE MAP!
  virtual void clear();

  /// Overwrites the layer with what's coming from the topic!
  void tsdfMapCallback(const voxblox_msgs::Layer& layer_msg);

  /// Update output filenames, used for multiple experiments of CRF
  void updateOutputFileNames();
  void saveeverything();

 protected:
  /**
   * Gets the next pointcloud that has an available transform to process from
   * the queue.
   */
  bool getNextPointcloudFromQueue(
      std::queue<sensor_msgs::PointCloud2::Ptr>* queue,
      sensor_msgs::PointCloud2::Ptr* pointcloud_msg, Transformation* T_G_C);

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  /// Data subscribers.
  ros::Subscriber pointcloud_sub_;
  ros::Subscriber freespace_pointcloud_sub_;

  /// Publish markers for visualization.
  ros::Publisher mesh_pub_;
  ros::Publisher tsdf_pointcloud_pub_;
  ros::Publisher surface_pointcloud_pub_;
  ros::Publisher tsdf_slice_pub_;
  ros::Publisher occupancy_marker_pub_;
  ros::Publisher icp_transform_pub_;
  ros::Publisher tsdf_voxel_pub_;

  /// Publish the complete map for other nodes to consume.
  ros::Publisher tsdf_map_pub_;

  // Publish the command to pc_node to pause
  ros::Publisher command_to_pc_publisher_pub_;

  /// Subscriber to subscribe to another node generating the map.
  ros::Subscriber tsdf_map_sub_;

  // Services.
  ros::ServiceServer generate_mesh_srv_;
  ros::ServiceServer generate_rgb_mesh_srv_;
  ros::ServiceServer generate_semantic_mesh_srv_;
  ros::ServiceServer generate_semantic_pc_srv_;
  ros::ServiceServer generate_semantic_probability_srv_;
  ros::ServiceServer display_tsdf_voxels;
  ros::ServiceServer clear_map_srv_;
  ros::ServiceServer save_map_srv_;
  ros::ServiceServer load_map_srv_;
  ros::ServiceServer publish_pointclouds_srv_;
  ros::ServiceServer publish_tsdf_map_srv_;

  /// Tools for broadcasting TFs.
  tf::TransformBroadcaster tf_broadcaster_;

  // Timers.
  ros::Timer update_mesh_timer_;
  ros::Timer publish_map_timer_;

  // parameters for semantic fusion
  bool verbose_;
  bool visualize_semantic;
  bool load_semantic_probability;
  bool load_geo_complexity;
  bool adaptive_mapping;
  int adaptive_ratio_small;   // must be an integer
  int adaptive_ratio_middle;  // must be an integer
  std::string semantic_update_method;
  bool fuse_semantic_for_subvoxel;

  // parameters for spatial consistency
  float theta_alpha;
  float theta_beta;
  float theta_gamma;
  float weight_appearance;
  float weight_smooth;

  // paremeters for saving map for multiple experiments
  std::string output_dir;
  std::string scene_name;
  int semantic_level_split_id;

  /**
   * Global/map coordinate frame. Will always look up TF transforms to this
   * frame.
   */
  std::string world_frame_;
  /**
   * Name of the ICP corrected frame. Publishes TF and transform topic to this
   * if ICP on.
   */
  std::string icp_corrected_frame_;
  /// Name of the pose in the ICP correct Frame.
  std::string pose_corrected_frame_;

  /// Delete blocks that are far from the system to help manage memory
  double max_block_distance_from_body_;

  /// Pointcloud visualization settings.
  double slice_level_;

  /// If the system should subscribe to a pointcloud giving points in freespace
  bool use_freespace_pointcloud_;

  /**
   * Mesh output settings. Mesh is only written to file if mesh_filename_ is
   * not empty.
   */
  std::string mesh_filename_;
  /// How to color the mesh.
  ColorMode color_mode_;

  /// Colormap to use for intensity pointclouds.
  std::shared_ptr<ColorMap> color_map_;

  /// Will throttle to this message rate.
  ros::Duration min_time_between_msgs_;

  /// What output information to publish
  bool publish_pointclouds_on_update_;
  bool publish_slices_;
  bool publish_pointclouds_;
  bool publish_tsdf_map_;

  /// Whether to save the latest mesh message sent (for inheriting classes).
  bool cache_mesh_;

  /**
   *Whether to enable ICP corrections. Every pointcloud coming in will attempt
   * to be matched up to the existing structure using ICP. Requires the initial
   * guess from odometry to already be very good.
   */
  bool enable_icp_;
  /**
   * If using ICP corrections, whether to store accumulate the corrected
   * transform. If this is set to false, the transform will reset every
   * iteration.
   */
  bool accumulate_icp_corrections_;

  /// Subscriber settings.
  int pointcloud_queue_size_;
  int num_subscribers_tsdf_map_;

  // Maps and integrators.
  std::shared_ptr<TsdfMap> tsdf_map_;
  std::unique_ptr<TsdfIntegratorBase> tsdf_integrator_;

  /// ICP matcher
  std::shared_ptr<ICP> icp_;

  // Mesh accessories.
  std::shared_ptr<MeshLayer> mesh_layer_;
  std::unique_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;
  /// Optionally cached mesh message.
  voxblox_msgs::Mesh cached_mesh_msg_;

  /**
   * Transformer object to keep track of either TF transforms or messages from
   * a transform topic.
   */
  Transformer transformer_;
  /**
   * Queue of incoming pointclouds, in case the transforms can't be immediately
   * resolved.
   */
  std::queue<sensor_msgs::PointCloud2::Ptr> pointcloud_queue_;
  std::queue<sensor_msgs::PointCloud2::Ptr> freespace_pointcloud_queue_;

  // Last message times for throttling input.
  ros::Time last_msg_time_ptcloud_;
  ros::Time last_msg_time_freespace_ptcloud_;

  /// Current transform corrections from ICP.
  Transformation icp_corrected_transform_;
};

}  // namespace voxblox

#endif  // ADAPTIVE_VOXEL_SERVER_H_
