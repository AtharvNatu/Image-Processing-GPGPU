#pragma once

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <omp.h>

#include "../Common/Macros.hpp"

#ifndef _HELPER_TIMER_H_
    #define _HELPER_TIMER_H_
    #include "../Common/helper_timer.h"
#endif

#include "ImageUtils.hpp"

class OtsuBinarizerCPU
{
    private:
        StopWatchInterface *cpuTimer = nullptr;

    public:

        OtsuBinarizerCPU(void);
        ~OtsuBinarizerCPU();
        
        /// @brief Generate Histogram From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @param pixelCount Total Pixels in image
        /// @param cpuTime [OUT] CPU Execution Time For Computing Histogram
        /// @return STL Vector containing histogram values
        std::vector<double> computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, bool multiThreading, int threadCount, size_t* pixelCount, double *cpuTime);


        /// @brief Get Threshold From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @param cpuTime [OUT] CPU Execution Time For Computing Threshold
        /// @return Integer threshold value for the image
        int computeThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, bool multiThreading, int threadCount, double *cpuTime);
};

