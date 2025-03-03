#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/image_transport.h>
#include <map_adapt_ros/pcl_type.h>
#include <nav_msgs/Path.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <signal.h>
#include <std_msgs/String.h>
#include <voxblox/core/common.h>

#include <Eigen/Geometry>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

void MySigintHandler(int sig) {
  ROS_INFO("shutting down!");
  ros::shutdown();
  exit(0);
}

class PCCommandListener {
 public:
  PCCommandListener(bool *if_pause) : if_pause_(if_pause) {}

  void ProcessCommand(const std_msgs::String::ConstPtr &command) {
    if (command->data == "pause") {
      *if_pause_ = true;
    } else if (command->data == "resume") {
      *if_pause_ = false;
    }
  }

  bool *if_pause_;
};

void LoadPoses(const string &strPathToTxt, vector<double> &vTimstamps,
               vector<Eigen::Isometry3d,
                      Eigen::aligned_allocator<Eigen::Isometry3d>> &vPoses);

geometry_msgs::PoseStamped GenerateVisualizePose(
    Eigen::Isometry3d &TransformationVisual, Eigen::Isometry3d &CameraPose,
    const float &TimeStamp);

geometry_msgs::TransformStamped GenerateVisualizeTransform(
    Eigen::Isometry3d &TransformationVisual, int &idx,
    Eigen::Isometry3d &CameraPose, const float &TimeStamp);

int main(int argc, char **argv) {
  ros::init(argc, argv, "pub_pc_semantic");

  ros::NodeHandle nh("~");
  ros::Publisher pcl_pub =
      nh.advertise<sensor_msgs::PointCloud2>("pcl_output", 100);
  ros::Publisher pose_pub =
      nh.advertise<geometry_msgs::PoseStamped>("pose_output", 100);
  ros::Publisher tf_pub =
      nh.advertise<geometry_msgs::TransformStamped>("tf_output", 100);
  ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("trajectory", 100);
  ros::Publisher path_gt_pub =
      nh.advertise<nav_msgs::Path>("trajectory_gt", 100);
  ros::Publisher end_command_pub =
      nh.advertise<std_msgs::String>("commander", 1);
  image_transport::ImageTransport it(nh);
  image_transport::Publisher image_pub = it.advertise("image", 100);
  sensor_msgs::PointCloud2 PointCloudCurrent;
  PointCloudCurrent.header.frame_id = "world";

  bool pause_publish_pc = false;
  PCCommandListener command_listener(&pause_publish_pc);
  ros::Subscriber pause_command_sub =
      nh.subscribe("pub_pc_switch", 5, &PCCommandListener::ProcessCommand,
                   &command_listener);

  /* Load parameters*/
  string root_dir;
  string path_to_estimated_pose;
  double rate = 10;
  double wait_time = 0.1;
  bool use_gt = false;
  double fx = 320.0;
  double fy = 240.0;
  double cx = 319.0;
  double cy = 239.0;
  nh.param("root_dir", root_dir, root_dir);
  nh.param("path_to_estimated_pose", path_to_estimated_pose,
           path_to_estimated_pose);
  nh.param("rate", rate, rate);
  nh.param("wait_time", wait_time, wait_time);
  nh.param("use_gt", use_gt, use_gt);
  nh.param("fx", fx, fx);
  nh.param("fy", fy, fy);
  nh.param("cx", cx, cx);
  nh.param("cy", cy, cy);

  path_to_estimated_pose = root_dir + path_to_estimated_pose;

  string path_to_label = root_dir + "/top4_labels/";
  if (use_gt) {
    path_to_label = root_dir + "/semantics/";
  }

  string path_to_probability = root_dir + "top4_probability/";

  cout << "ROS rate is: " << rate << endl;
  cout << "ROS wait_time is: " << wait_time << endl;
  cout << "Path to the root folder is: " << root_dir << endl;
  cout << "Path to the estimated pose file is: " << path_to_estimated_pose
       << endl;
  cout << "Path to the folder of semantic labels is: " << path_to_label << endl;
  cout << "Path to the folder of semantic probabilities is: "
       << path_to_probability << endl;

  vector<double> TimeStamps;
  vector<double> TimeStamps_gt;
  vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
  vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
      poses_gt;
  nav_msgs::Path path;
  path.header.frame_id = "world";
  nav_msgs::Path path_gt;
  path_gt.header.frame_id = "world";

  Eigen::Quaterniond q_visual(-0.5, 0.5, -0.5, 0.5);
  Eigen::Isometry3d T_visual(q_visual);
  T_visual.translate(Eigen::Vector3d(0.0, -1.0, 0.0));

  std::cout << path_to_estimated_pose << std::endl;

  LoadPoses(path_to_estimated_pose, TimeStamps, poses);
  LoadPoses(root_dir + "/gt_pose_calibrated.txt", TimeStamps_gt, poses_gt);

  int idx = 0;
  ros::Rate loop_rate(rate);

  ros::Duration(wait_time).sleep();

  std::cout << "Start to publish" << std::endl;

  signal(SIGINT, MySigintHandler);
  while (ros::ok) {
    ros::spinOnce();

    if (idx % TimeStamps.size() == 0) {
      path.poses.clear();
      path_gt.poses.clear();
      if (idx != 0) {
        ros::Duration(0.5).sleep();
        std_msgs::String end_command;
        end_command.data = "End the system";
        end_command_pub.publish(end_command);
        break;
      }
    }

    geometry_msgs::PoseStamped PoseStampedEst, PoseStampedGt;
    geometry_msgs::TransformStamped TFStampedEst;
    TFStampedEst =
        GenerateVisualizeTransform(T_visual, idx, poses[idx], TimeStamps[idx]);
    TFStampedEst.header.stamp = ros::Time(TimeStamps[idx]);
    PoseStampedEst =
        GenerateVisualizePose(T_visual, poses[idx], TimeStamps[idx]);
    PoseStampedGt =
        GenerateVisualizePose(T_visual, poses_gt[idx], TimeStamps_gt[idx]);
    path.header.stamp = ros::Time(TimeStamps[idx]);
    path_gt.header.stamp = ros::Time(TimeStamps_gt[idx]);

    // Before publish, check if we are required to pause here
    if (pause_publish_pc) continue;

    path.poses.push_back(PoseStampedEst);
    path_gt.poses.push_back(PoseStampedGt);

    pose_pub.publish(PoseStampedEst);
    path_pub.publish(path);
    path_gt_pub.publish(path_gt);
    tf_pub.publish(TFStampedEst);

    cv::Mat imRGB, imD, imgGeo, imgSemantics, imgProbability;
    string niStr = to_string(idx);
    int str_length = niStr.length();
    for (int pad_i = 0; pad_i < 5 - str_length; pad_i++) niStr = "0" + niStr;

    imRGB = cv::imread(root_dir + "/rgb/frame_" + niStr + ".png",
                       cv::IMREAD_UNCHANGED);

    imD = cv::imread(root_dir + "/depth/frame_" + niStr + ".TIFF",
                     cv::IMREAD_UNCHANGED);

    imgGeo = cv::imread(
        root_dir + "/change_of_curvature_stride_2/frame_" + niStr + ".TIFF",
        cv::IMREAD_UNCHANGED);

    imgSemantics = cv::imread(path_to_label + "frame_" + niStr + ".png",
                              cv::IMREAD_UNCHANGED);
    if (!use_gt) {
      imgProbability =
          cv::imread(path_to_probability + "frame_" + niStr + ".png",
                     cv::IMREAD_UNCHANGED);
    }

    pcl::PointCloud<PointXYZRGBILabelMap> pointCloud;
    pointCloud.points.clear();
    pointCloud.is_dense = false;
    for (int u = 0; u < imRGB.cols; u += 1) {
      for (int v = 0; v < imRGB.rows; v += 1) {
        float d = imD.at<float>(v, u);
        if (d <= 0) continue;
        // if (d > 10) continue;
        Eigen::Vector3d point;
        point[2] = double(d);
        point[0] = (u - cx) * point[2] / fx;
        point[1] = (v - cy) * point[2] / fy;

        PointXYZRGBILabelMap p;
        p.x = point[0];
        p.y = point[1];
        p.z = point[2];
        p.b = imRGB.at<cv::Vec3b>(v, u)[0];
        p.g = imRGB.at<cv::Vec3b>(v, u)[1];
        p.r = imRGB.at<cv::Vec3b>(v, u)[2];

        p.intensity = imgGeo.at<float>(v, u);
        if (p.intensity > 0.33333f) {
          std::cerr << "wrong intensity: " << p.intensity << std::endl;
        }

        if (use_gt) {
          p.top4_label =
              static_cast<uint32_t>(imgSemantics.at<uchar>(v, u) + 1);
          p.top4_probability = static_cast<uint32_t>(1.0 * 255.0);
        } else {
          p.top4_label =
              static_cast<uint32_t>(imgSemantics.at<cv::Vec4b>(v, u)[0]) |
              (static_cast<uint32_t>(imgSemantics.at<cv::Vec4b>(v, u)[1])
               << 8) |
              (static_cast<uint32_t>(imgSemantics.at<cv::Vec4b>(v, u)[2])
               << 16) |
              (static_cast<uint32_t>(imgSemantics.at<cv::Vec4b>(v, u)[3])
               << 24);
          p.top4_probability =
              static_cast<uint32_t>(imgProbability.at<cv::Vec4b>(v, u)[0]) |
              (static_cast<uint32_t>(imgProbability.at<cv::Vec4b>(v, u)[1])
               << 8) |
              (static_cast<uint32_t>(imgProbability.at<cv::Vec4b>(v, u)[2])
               << 16) |
              (static_cast<uint32_t>(imgProbability.at<cv::Vec4b>(v, u)[3])
               << 24);
        }

        pointCloud.points.push_back(p);
      }
    }

    pcl::toROSMsg(pointCloud, PointCloudCurrent);
    PointCloudCurrent.header.frame_id = "world";
    PointCloudCurrent.header.stamp = ros::Time(TimeStamps[idx]);
    pcl_pub.publish(PointCloudCurrent);

    // sensor_msgs::ImagePtr imageMsg =
    //     cv_bridge::CvImage(std_msgs::Header(), "bgr8", imRGB).toImageMsg();
    // imageMsg->header.frame_id = "world";
    // image_pub.publish(imageMsg);

    loop_rate.sleep();

    idx += 1;
  }

  return 0;
}

void LoadPoses(const string &strPathToTxt, vector<double> &vTimstamps,
               vector<Eigen::Isometry3d,
                      Eigen::aligned_allocator<Eigen::Isometry3d>> &vPoses) {
  ifstream fTraj;
  fTraj.open(strPathToTxt.c_str());

  while (!fTraj.eof()) {
    string s;
    getline(fTraj, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimstamps.push_back(t);

      double tx, ty, tz;
      double qx, qy, qz, qw;

      ss >> tx;
      ss >> ty;
      ss >> tz;
      ss >> qx;
      ss >> qy;
      ss >> qz;
      ss >> qw;

      Eigen::Quaterniond q(qw, qx, qy, qz);
      Eigen::Isometry3d T(q);
      T.pretranslate(Eigen::Vector3d(tx, ty, tz));
      vPoses.push_back(T);
    }
  }
}

geometry_msgs::PoseStamped GenerateVisualizePose(
    Eigen::Isometry3d &TransformationVisual, Eigen::Isometry3d &CameraPose,
    const float &TimeStamp) {
  geometry_msgs::PoseStamped poseStamped;
  poseStamped.header.frame_id = "world";
  Eigen::Isometry3d pose_visual = TransformationVisual * CameraPose;
  Eigen::Quaterniond q_visual(pose_visual.rotation());
  Eigen::Matrix4d M_visual = pose_visual.matrix();
  poseStamped.pose.position.x = M_visual(0, 3);
  poseStamped.pose.position.y = M_visual(1, 3);
  poseStamped.pose.position.z = M_visual(2, 3);
  poseStamped.pose.orientation.x = q_visual.x();
  poseStamped.pose.orientation.y = q_visual.y();
  poseStamped.pose.orientation.z = q_visual.z();
  poseStamped.pose.orientation.w = q_visual.w();
  poseStamped.header.stamp = ros::Time(TimeStamp);
  poseStamped.header.frame_id = "world";

  return poseStamped;
}

geometry_msgs::TransformStamped GenerateVisualizeTransform(
    Eigen::Isometry3d &TransformationVisual, int &idx,
    Eigen::Isometry3d &CameraPose, const float &TimeStamp) {
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.frame_id = "world";
  transformStamped.header.seq = idx;
  transformStamped.header.stamp = ros::Time(TimeStamp);
  transformStamped.child_frame_id = "laser";
  Eigen::Isometry3d pose_visual = TransformationVisual * CameraPose;
  Eigen::Quaterniond q_visual(pose_visual.rotation());
  Eigen::Matrix4d M_visual = pose_visual.matrix();
  transformStamped.transform.translation.x = M_visual(0, 3);
  transformStamped.transform.translation.y = M_visual(1, 3);
  transformStamped.transform.translation.z = M_visual(2, 3);
  transformStamped.transform.rotation.x = q_visual.x();
  transformStamped.transform.rotation.y = q_visual.y();
  transformStamped.transform.rotation.z = q_visual.z();
  transformStamped.transform.rotation.w = q_visual.w();

  return transformStamped;
}