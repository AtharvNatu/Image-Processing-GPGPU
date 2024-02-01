#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "../Common/Macros.hpp"
#include "../Common/Logger.hpp"

using namespace std;

class ImageUtils
{
    public:
        // Image Functions
        cv::Mat loadImage(string imagePath);
        void saveImage(string imagePath, cv::Mat* image);
};

