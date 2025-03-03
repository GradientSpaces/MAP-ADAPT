#ifndef MAP_ADAPT_ROS_INTERACTIVE_SLIDER_H_
#define MAP_ADAPT_ROS_INTERACTIVE_SLIDER_H_

#include <interactive_markers/interactive_marker_server.h>
#include <visualization_msgs/InteractiveMarkerFeedback.h>
#include <voxblox/core/common.h>

#include <functional>
#include <string>

namespace voxblox {

/// InteractiveSlider class which can be used for visualizing voxel map slices.
class InteractiveSlider {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  InteractiveSlider(
      const std::string& slider_name,
      const std::function<void(const double& slice_level)>& slider_callback,
      const Point& initial_position, const unsigned int free_plane_index,
      const float marker_scale_meters);
  virtual ~InteractiveSlider() {}

 private:
  const unsigned int free_plane_index_;
  interactive_markers::InteractiveMarkerServer interactive_marker_server_;

  /// Processes the feedback after moving the slider.
  virtual void interactiveMarkerFeedback(
      const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback,
      const std::function<void(const double slice_level)>& slider_callback);
};

}  // namespace voxblox

#endif  // MAP_ADAPT_ROS_INTERACTIVE_SLIDER_H_
