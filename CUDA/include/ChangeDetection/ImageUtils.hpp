#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "../Common/Macros.hpp"
#include "../Common/Logger.hpp"

class ImageUtils
{
    public:
        // Image Functions

        /// @brief Function to load image from disk
        /// @param imagePath Absolute image path from where image will be loaded
        /// @return Image in cv::Mat format
        cv::Mat loadImage(std::string imagePath);

        /// @brief Function to save the image on disk
        /// @param imagePath Absolute image path where image will be saved
        /// @param image Pointer to cv::Mat Image
        void saveImage(std::string imagePath, cv::Mat* image);

        /// @brief Get Image Raw Pixel Data
        /// @param image Input Image
        /// @return Vector containing raw pixel data
        std::vector<uchar_t> getRawPixelData(cv::Mat* image);
};

