#pragma once

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "ImageUtils.hpp"
#include "../Common/Macros.hpp"
#include "../Common/CudaUtils.cuh"

class OtsuBinarizerCuda
{
    private:
        ImageUtils *imageUtils = nullptr;
        
        uint_t *hostHistogram = nullptr, *deviceHistogram = nullptr;
        uchar_t *devicePixelData = nullptr;
        double *normalizedHistogram = nullptr;

    public:

        OtsuBinarizerCuda(void);
        ~OtsuBinarizerCuda(void);

        
        /// @brief Generate Histogram From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @param pixelCount Total Pixels in image
        /// @return STL Vector containing histogram values
        double* computeHistogram(cv::Mat* inputImage, long *pixelCount);


        /// @brief Get Threshold From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @return Integer threshold value for the image
        int computeThreshold(cv::Mat* inputImage, bool multiThreading, int threadCount);
};

//* CUDA Kernel Prototypes
__global__ void cudaHistogram(uchar_t *pixelData, uint_t *histogram, long segmentSize, long totalPixels);

__global__ void cudaComputeClassVariances(double *histogram, double allProbabilitySum, long totalPixels, double *betweenClassVariances);
