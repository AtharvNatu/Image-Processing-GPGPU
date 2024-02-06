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
        cv::Mat loadImage(std::string imagePath);
        void saveImage(std::string imagePath, cv::Mat* image);
};

