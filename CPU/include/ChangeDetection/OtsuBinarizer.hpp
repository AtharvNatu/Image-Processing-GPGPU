#pragma once

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <omp.h>

#include "../Common/Macros.hpp"

class OtsuBinarizerCPU
{
    public:
        
        /// @brief Generate Histogram From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @param pixelCount Total Pixels in image
        /// @return STL Vector containing histogram values
        std::vector<double> computeHistogram(cv::Mat* inputImage, bool multiThreading, int threadCount, size_t* pixelCount);


        /// @brief Get Threshold From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @return Integer threshold value for the image
        int computeThreshold(cv::Mat* inputImage, bool multiThreading, int threadCount);
};

