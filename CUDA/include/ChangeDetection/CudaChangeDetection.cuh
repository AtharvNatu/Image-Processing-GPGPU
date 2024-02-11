#pragma once

#if (OS == 1)
    #include <windows.h>
#endif

#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "../Common/Macros.hpp"
#include "../Common/Logger.hpp"
#include "../Common/CudaUtils.cuh"

#ifndef _HELPER_TIMER_H_
    #define _HELPER_TIMER_H_
    #include "../Common/helper_timer.h"
#endif

#include "ImageUtils.hpp"
#include "OtsuBinarizerCuda.cuh"


class CudaChangeDetection
{
    // Member Variables
    private:
        StopWatchInterface *cudaTimer = nullptr;
        ImageUtils *imageUtils = nullptr;
        OtsuBinarizerCuda *binarizer = nullptr;

        uchar3 *hostOldImage = nullptr, *hostNewImage = nullptr, *hostOutputImage = nullptr;
        uchar3 *deviceOldImage = nullptr, *deviceNewImage = nullptr, *deviceOutputImage = nullptr;

    public:
        Logger *logger = nullptr;

    // Member Functions
    public:

        //* DEBUG Mode
        CudaChangeDetection(void);

        //* RELEASE Mode
        CudaChangeDetection(std::string logFilePath);

        ~CudaChangeDetection();

        /// @brief Change Detection Wrapper Function
        /// @param oldImagePath Image with old timestamp
        /// @param newImagePath Image with new timestamp
        /// @param outputPath Directory path to store output image
        /// @param grayscale  Output Image format : Grayscale (RED Color Changes) or Binary (WHITE Color Changes)
        /// @return Time required for the kernel to execute on GPU in seconds
        double detectChanges(std::string oldImagePath, std::string newImagePath, std::string outputPath, bool grayscale);

        /// @brief Memory Cleanup For Host and Device
        /// @param None
        void cleanup(void);
};

//* CUDA Kernel Prototypes

// @brief CUDA Grayscale Change Detection Kernel | Output Image format : Grayscale (RED Color Changes)
/// @param oldImage Old Image Pixel Array in uchar3 Format
/// @param newImage New Image Pixel Array in uchar3 Format
/// @param outputImage Output Image Pixel Array (Blank) in uchar3 Format
/// @param threshold  Threshold computed using Otsu Binarizer
/// @param size  Size of Old Image Pixel Array
__global__ void grayscaleChangeDetection(uchar3 *oldImage, uchar3 *newImage, uchar3 *outputImage, int threshold, size_t size);

// @brief CUDA Binary Change Detection Kernel | Output Image format : Binary (WHITE Color Changes)
/// @param oldImage Old Image Pixel Array in uchar3 Format
/// @param newImage New Image Pixel Array in uchar3 Format
/// @param outputImage Output Image Pixel Array (Blank) in uchar3 Format
/// @param threshold  Threshold computed using Otsu Binarizer
/// @param size  Size of Old Image Pixel Array
__global__ void binaryChangeDetection(uchar3 *oldImage, uchar3 *newImage, uchar3 *outputImage, int threshold, size_t size);




