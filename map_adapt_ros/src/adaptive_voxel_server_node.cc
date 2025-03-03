#include <gflags/gflags.h>
#include <std_msgs/String.h>

#include "map_adapt_ros/adaptive_voxel_server.h"

class CommandListener {
 public:
  CommandListener(bool* if_continue) : if_continue_(if_continue) {}

  void ProcessCommand(const std_msgs::String command) {
    std::cout << "time to stop" << std::endl;
    *if_continue_ = false;
  }

  bool* if_continue_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "voxblox");
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  voxblox::VoxelServer node(nh, nh_private);

  bool continue_process = true;
  CommandListener listener(&continue_process);

  /// End command subscribers.
  ros::Subscriber command_sub =
      nh.subscribe("commander", 1, &CommandListener::ProcessCommand, &listener);

  while (ros::ok()) {
    if (!continue_process) {
      node.updateOutputFileNames();
      node.generateMesh();
      node.saveeverything();

      break;
    }
    ros::spinOnce();
  }
  return 0;
}
