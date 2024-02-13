#pragma once

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "../Common/Macros.hpp"
#include "../Common/CudaUtils.cuh"

#ifndef _HELPER_TIMER_H_
    #define _HELPER_TIMER_H_
    #include "../Common/helper_timer.h"
#endif

#include "ImageUtils.hpp"

class OtsuBinarizerCuda
{
    private:
        StopWatchInterface *gpuTimer = nullptr;

    public:

        OtsuBinarizerCuda(void);
        ~OtsuBinarizerCuda();

        /// @brief Generate Histogram From Input Image
        /// @param inputImage [IN] cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param pixelCount [OUT] Total Pixels in image
        /// @param gpuTime [OUT] Kernel Execution Time
        /// @return Array containing histogram values in double precision
        double* computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, size_t *pixelCount, double *gpuTime);


        /// @brief Get Threshold From Input Image
        /// @param inputImage [IN] cv::Mat Pointer to input image
        /// @param imageUtils Instance of ImageUtils Class
        /// @param gpuTime [OUT] Kernel Execution Time
        /// @return Integer threshold value for the image
        int computeThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, double *gpuTime);
};

//* CUDA Kernel Prototypes
__global__ void cudaHistogram(uchar_t *pixelData, uint_t *histogram, int segmentSize, size_t totalPixels);

__global__ void cudaComputeClassVariances(double *histogram, double allProbabilitySum, double *betweenClassVariances, size_t totalPixels);
