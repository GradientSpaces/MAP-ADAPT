#ifndef _PCL_TYPE_H_
#define _PCL_TYPE_H_

#include <glog/logging.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

struct PointXYZRGBLabelMap {
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  std::uint32_t top4_label;
  std::uint32_t top4_probability;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZRGBLabelMap,
    (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(
        std::uint32_t, top4_label, top4_label)(std::uint32_t, top4_probability,
                                               top4_probability))

struct PointXYZRGBI {
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  float intensity;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZRGBI,
    (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(float, intensity,
                                                             intensity))

struct PointXYZRGBNORMALs {
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  float n_x;
  float n_y;
  float n_z;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZRGBNORMALs,
    (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(float, n_x, n_x)(
        float, n_y, n_y)(float, n_z, n_z))

struct PointXYZRGBLabelNormal {
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  std::uint32_t top4_label;
  std::uint32_t top4_probability;
  float n_x;
  float n_y;
  float n_z;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZRGBLabelNormal,
    (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(std::uint32_t,
                                                             top4_label,
                                                             top4_label)(
        std::uint32_t, top4_probability,
        top4_probability)(float, n_x, n_x)(float, n_y, n_y)(float, n_z, n_z))

struct PointXYZRGBILabelMap {
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  float intensity;
  std::uint32_t top4_label;
  std::uint32_t top4_probability;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZRGBILabelMap,
    (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(float, intensity,
                                                             intensity)(
        std::uint32_t, top4_label, top4_label)(std::uint32_t, top4_probability,
                                               top4_probability))

#endif