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

class OtsuThresholdCPU
{
    private:
        StopWatchInterface *cpuTimer = nullptr;

        /// @brief Generate Histogram From Input Image - Single-threaded
        /// @param inputImage cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param pixelCount Total Pixels in image
        /// @param cpuTime [OUT] CPU Execution Time For Computing Histogram
        /// @return STL Vector containing histogram values
        std::vector<double> computeHistogramST(cv::Mat* inputImage, ImageUtils *imageUtils, size_t* pixelCount, double *cpuTime);
        
        /// @brief Generate Histogram From Input Image - Multi-threaded
        /// @param inputImage cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param pixelCount Total Pixels in image
        /// @param cpuTime [OUT] CPU Execution Time For Computing Histogram
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @return STL Vector containing histogram values
        std::vector<double> computeHistogramMT(cv::Mat* inputImage, ImageUtils *imageUtils, size_t* pixelCount, double *cpuTime, int threadCount);

        /// @brief Get Threshold From Input Image - Multi-threaded
        /// @param inputImage cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param cpuTime [OUT] CPU Execution Time For Computing Threshold
        /// @return Integer threshold value for the image
        int computeThresholdST(cv::Mat* inputImage, ImageUtils *imageUtils, double *cpuTime);

        /// @brief Get Threshold From Input Image - Single-threaded
        /// @param inputImage cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param cpuTime [OUT] CPU Execution Time For Computing Threshold
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @return Integer threshold value for the image
        int computeThresholdMT(cv::Mat* inputImage, ImageUtils *imageUtils, double *cpuTime, int threadCount);

    public:

        OtsuThresholdCPU(void);
        ~OtsuThresholdCPU();

        /// @brief Get Threshold From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @param cpuTime [OUT] CPU Execution Time For Computing Threshold
        /// @return Integer threshold value for the image
        int getImageThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, bool multiThreading, int threadCount, double *cpuTime);
};

