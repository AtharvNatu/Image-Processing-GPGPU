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

constexpr int HIST_BINS = 256;

class OtsuThresholdCuda
{
    private:
        //* Serial Timer
        StopWatchInterface *gpuTimer = nullptr;

        //* Parallel Timer
        cudaEvent_t start, end;

    public:

        OtsuThresholdCuda(void);
        ~OtsuThresholdCuda();

        /// @brief Generate Histogram From Input Image
        /// @param inputImage [IN] cv::Mat Pointer to input image
        /// @param pixelCount [OUT] Total Pixels in image
        /// @param gpuTime [OUT] Kernel Execution Time
        /// @param imageUtils Instance of ImageUtils Class
        /// @param cudaUtils Instance of CudaUtils Class
        /// @return Array containing histogram values in double precision
        double* computeHistogram(cv::Mat* inputImage, size_t *pixelCount, double *gpuTime, ImageUtils *imageUtils, CudaUtils *cudaUtils);


        /// @brief Get Threshold From Input Image
        /// @param inputImage [IN] cv::Mat Pointer to input image
        /// @param gpuTime [OUT] Kernel Execution Time
        /// @param imageUtils Instance of ImageUtils Class
        /// @param cudaUtils Instance of CudaUtils Class
        /// @return Integer threshold value for the image
        int computeThreshold(cv::Mat* inputImage, double *gpuTime, ImageUtils *imageUtils, CudaUtils *cudaUtils);
};

//* CUDA Kernel Prototypes
__global__ void cudaHistogram(uchar_t *pixelData, uint_t *histogram, int segmentSize, size_t totalPixels);

__global__ void cudaComputeClassVariances(double *histogram, double allProbabilitySum, double *betweenClassVariances, size_t totalPixels);
