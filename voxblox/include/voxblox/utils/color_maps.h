#ifndef VOXBLOX_UTILS_COLOR_MAPS_H_
#define VOXBLOX_UTILS_COLOR_MAPS_H_

#include <algorithm>
#include <vector>

#include "voxblox/core/color.h"
#include "voxblox/core/common.h"

namespace voxblox {

class ColorMap {
 public:
  ColorMap() : min_value_(0.0), max_value_(1.0) {}
  virtual ~ColorMap() {}

  void setMinValue(float min_value) { min_value_ = min_value; }

  void setMaxValue(float max_value) { max_value_ = max_value; }

  virtual Color colorLookup(float value) const = 0;

 protected:
  float min_value_;
  float max_value_;
};

class GrayscaleColorMap : public ColorMap {
 public:
  virtual Color colorLookup(float value) const {
    float new_value = std::min(max_value_, std::max(min_value_, value));
    new_value = (new_value - min_value_) / (max_value_ - min_value_);
    return grayColorMap(new_value);
  }
};

class InverseGrayscaleColorMap : public ColorMap {
 public:
  virtual Color colorLookup(float value) const {
    float new_value = std::min(max_value_, std::max(min_value_, value));
    new_value = (new_value - min_value_) / (max_value_ - min_value_);
    return grayColorMap(1.0 - new_value);
  }
};

class RainbowColorMap : public ColorMap {
 public:
  virtual Color colorLookup(float value) const {
    float new_value = std::min(max_value_, std::max(min_value_, value));
    new_value = (new_value - min_value_) / (max_value_ - min_value_);
    return rainbowColorMap(new_value);
  }
};

class InverseRainbowColorMap : public ColorMap {
 public:
  virtual Color colorLookup(float value) const {
    float new_value = std::min(max_value_, std::max(min_value_, value));
    new_value = (new_value - min_value_) / (max_value_ - min_value_);
    return rainbowColorMap(1.0 - new_value);
  }
};

class IronbowColorMap : public ColorMap {
 public:
  IronbowColorMap() : ColorMap() {
    palette_colors_.push_back(Color(0, 0, 0));
    palette_colors_.push_back(Color(145, 20, 145));
    palette_colors_.push_back(Color(255, 138, 0));
    palette_colors_.push_back(Color(255, 230, 40));
    palette_colors_.push_back(Color(255, 255, 255));
    // Add an extra to avoid overflow.
    palette_colors_.push_back(Color(255, 255, 255));

    increment_ = 1.0 / (palette_colors_.size() - 2);
  }

  virtual Color colorLookup(float value) const {
    float new_value = std::min(max_value_, std::max(min_value_, value));
    new_value = (new_value - min_value_) / (max_value_ - min_value_);

    size_t index = static_cast<size_t>(std::floor(new_value / increment_));

    return Color::blendTwoColors(
        palette_colors_[index], increment_ * (index + 1) - new_value,
        palette_colors_[index + 1], new_value - increment_ * (index));
  }

 protected:
  std::vector<Color> palette_colors_;
  float increment_;
};

/* NYU color map*/
class NYUColorMap : public ColorMap {
 public:
  NYUColorMap() : ColorMap() {
    color_map_list.push_back(Color(0, 0, 0));
    color_map_list.push_back(Color(128, 0, 0));
    color_map_list.push_back(Color(0, 128, 0));
    color_map_list.push_back(Color(128, 128, 0));
    color_map_list.push_back(Color(0, 0, 128));
    color_map_list.push_back(Color(128, 0, 128));
    color_map_list.push_back(Color(0, 128, 128));
    color_map_list.push_back(Color(128, 128, 128));
    color_map_list.push_back(Color(64, 0, 0));
    color_map_list.push_back(Color(192, 0, 0));
    color_map_list.push_back(Color(64, 128, 0));
    color_map_list.push_back(Color(192, 128, 0));
    color_map_list.push_back(Color(64, 0, 128));
    color_map_list.push_back(Color(192, 0, 128));
    color_map_list.push_back(Color(64, 128, 128));
    color_map_list.push_back(Color(192, 128, 128));
    color_map_list.push_back(Color(0, 64, 0));
    color_map_list.push_back(Color(128, 64, 0));
    color_map_list.push_back(Color(0, 192, 0));
    color_map_list.push_back(Color(128, 192, 0));
    color_map_list.push_back(Color(0, 64, 128));
    color_map_list.push_back(Color(128, 64, 128));
    color_map_list.push_back(Color(0, 192, 128));
    color_map_list.push_back(Color(128, 192, 128));
    color_map_list.push_back(Color(64, 64, 0));
    color_map_list.push_back(Color(192, 64, 0));
    color_map_list.push_back(Color(64, 192, 0));
    color_map_list.push_back(Color(192, 192, 0));
    color_map_list.push_back(Color(64, 64, 128));
    color_map_list.push_back(Color(192, 64, 128));
    color_map_list.push_back(Color(64, 192, 128));
    color_map_list.push_back(Color(192, 192, 128));
    color_map_list.push_back(Color(0, 0, 64));
    color_map_list.push_back(Color(128, 0, 64));
    color_map_list.push_back(Color(0, 128, 64));
    color_map_list.push_back(Color(128, 128, 64));
    color_map_list.push_back(Color(0, 0, 192));
    color_map_list.push_back(Color(128, 0, 192));
    color_map_list.push_back(Color(0, 128, 192));
    color_map_list.push_back(Color(128, 128, 192));
  }

  virtual Color colorLookup(float value) const {
    if (ceil(value) != floor(value)) {
      std::cout << "*******Invalid label!! NYU colormap only accept integer "
                   "input, output black color*****"
                << std::endl;
      return color_map_list[0];
    }

    int label_id = (int)value;
    if (label_id > 40 || label_id < 1) {
      std::cout << "*******Invalid label!! NYU colormap only accept label from "
                   "1 to 40, output black color*****"
                << std::endl;
      return color_map_list[0];
    }
    // label starts from one, but index starts from 0
    return color_map_list[label_id - 1];
  }

 private:
  std::vector<Color> color_map_list;
};

// The followings are colormap used to visualize in MAP-ADAPT paper
class CustomColorMap : public ColorMap {
 public:
  CustomColorMap() : ColorMap() {
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(128, 255, 128));  // 2
    color_map_list.push_back(Color(255, 64, 64));    // 3
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 255, 64));    // 6
    color_map_list.push_back(Color(223, 0, 0));      // 7
    color_map_list.push_back(Color(255, 128, 128));  // 8
    color_map_list.push_back(Color(128, 0, 0));      // 9
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(0, 223, 0));  // 14
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));  // 17 !!!
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(0, 64, 128));  // 20 !!
    color_map_list.push_back(Color(0, 128, 0));   // 21
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(64, 64, 255));
    color_map_list.push_back(Color(0, 0, 64));   // 26
    color_map_list.push_back(Color(0, 0, 223));  // 27
    color_map_list.push_back(Color(64, 64, 255));
  }

  virtual Color colorLookup(float value) const {
    if (ceil(value) != floor(value)) {
      std::cout << "*******Invalid label!! Custom colormap only accept integer "
                   "input, output black color*****"
                << std::endl;
      return color_map_list[0];
    }

    int label_id = (int)value;
    if (label_id > 29 || label_id < 1) {
      std::cout
          << "*******Invalid label!! Custom colormap only accept label from "
             "1 to 29, output black color*****"
          << std::endl;
      return color_map_list[0];
    }
    // label starts from one, but index starts from 0
    return color_map_list[label_id - 1];
  }

 private:
  std::vector<Color> color_map_list;
};

class IdColorMap {
 public:
  IdColorMap() {}
  virtual ~IdColorMap() {}

  virtual Color colorLookup(size_t value) const = 0;
};
/**
 * Map unique IDs from [0, Inf[ to unique colors that have constant (irrational)
 * spacing and high spread on the color wheel.
 * An important advantage of this color map is that the colors are independent
 * of the max ID value, e.g. previous colors don't change when adding new IDs.
 */
class IrrationalIdColorMap : IdColorMap {
 public:
  IrrationalIdColorMap() : irrational_base_(3.f * M_PI) {}

  void setIrrationalBase(float value) { irrational_base_ = value; }

  virtual Color colorLookup(const size_t value) const {
    const float normalized_color = std::fmod(value / irrational_base_, 1.f);
    return rainbowColorMap(normalized_color);
  }

 private:
  float irrational_base_;
};
/**
 * Map unique IDs from [0, Inf[ to unique colors
 * with piecewise constant spacing and higher spread on the color wheel.
 * An important advantage of this color map is that the colors are independent
 * of the max ID value, e.g. previous colors don't change when adding new IDs.
 */
class ExponentialOffsetIdColorMap : IdColorMap {
 public:
  ExponentialOffsetIdColorMap() : items_per_revolution_(10u) {}

  void setItemsPerRevolution(uint value) { items_per_revolution_ = value; }

  virtual Color colorLookup(const size_t value) const {
    const size_t revolution = value / items_per_revolution_;
    const float progress_along_revolution =
        std::fmod(value / items_per_revolution_, 1.f);
    // NOTE: std::modf could be used to simultaneously get the integer and
    //       fractional parts, but the above code is assumed to be more readable

    // Calculate the offset if appropriate
    float offset = 0;
    if (items_per_revolution_ < value + 1u) {
      const size_t current_episode = std::floor(std::log2(revolution));
      const size_t episode_start = std::exp2(current_episode);
      const size_t episode_num_subdivisions = episode_start;
      const size_t current_subdivision = revolution - episode_start;
      const float subdivision_step_size =
          1 / (items_per_revolution_ * 2 * episode_num_subdivisions);
      offset = (2 * current_subdivision + 1) * subdivision_step_size;
    }

    const float normalized_color = progress_along_revolution + offset;
    return rainbowColorMap(normalized_color);
  }

 private:
  float items_per_revolution_;
};

}  // namespace voxblox

#endif  // VOXBLOX_UTILS_COLOR_MAPS_H_
