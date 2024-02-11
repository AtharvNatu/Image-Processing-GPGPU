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

    public:

        OtsuBinarizerCuda(void);
        ~OtsuBinarizerCuda(void);

        
        /// @brief Generate Histogram From Input Image
        /// @param inputImage [IN] cv::Mat Pointer to input image
        /// @param pixelCount [OUT] Total Pixels in image
        /// @return Array containing histogram values in double precision
        double* computeHistogram(cv::Mat* inputImage, long *pixelCount);


        /// @brief Get Threshold From Input Image
        /// @param inputImage cv::Mat Pointer to input image
        /// @return Integer threshold value for the image
        int computeThreshold(cv::Mat* inputImage);
};

//* CUDA Kernel Prototypes
__global__ void cudaHistogram(uchar_t *pixelData, uint_t *histogram, long segmentSize, long totalPixels);

__global__ void cudaComputeClassVariances(double *histogram, double allProbabilitySum, long totalPixels, double *betweenClassVariances);
