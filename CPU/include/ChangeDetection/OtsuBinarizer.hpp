#pragma once

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <omp.h>

#include "../Common/Macros.hpp"

class OtsuBinarizer
{
    public:
        
        /// @brief Generate Histogram From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @return STL Vector containing histogram values
        std::vector<double> getHistogram(cv::Mat* inputImage);


        /// @brief Get Threshold From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @return Integer threshold value for the image
        int getThreshold(cv::Mat* inputImage);
        
        // void binarize(cv::Mat* inputImage);
};

