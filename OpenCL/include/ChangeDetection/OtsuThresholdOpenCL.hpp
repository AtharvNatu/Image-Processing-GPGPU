#pragma once

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "../Common/Macros.hpp"

#ifndef _HELPER_TIMER_H_
    #define _HELPER_TIMER_H_
    #include "../Common/helper_timer.h"
#endif

#include "../CLFW/CLFW.hpp"

#include "ImageUtils.hpp"

constexpr int HIST_BINS = 256;

class OtsuThresholdOpenCL
{
    private:
        StopWatchInterface *gpuTimer = nullptr;
        
    public:

        OtsuThresholdOpenCL(void);
        ~OtsuThresholdOpenCL();
        
        /// @brief Generate Histogram From Input Image
        /// @param inputImage [IN] cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param pixelCount [OUT] Total Pixels in image
        /// @param gpuTime [OUT] Kernel Execution Time
        /// @return Array containing histogram values in double precision
        
        //? double* computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, size_t *pixelCount, double *gpuTime);
        //? void computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, size_t *pixelCount, double *gpuTime);
        
        //! Placeholder
        std::vector<double> computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, size_t *pixelCount, double *gpuTime);


        /// @brief Get Threshold From Input Image
        /// @param inputImage [IN] cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param gpuTime [OUT] Kernel Execution Time
        /// @return Integer threshold value for the image
        int computeThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, double *gpuTime);
};

