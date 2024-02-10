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
#include "../Common/helper_timer.h"

#include "ImageUtils.hpp"
#include "OtsuBinarizerCuda.cuh"


class CudaChangeDetection
{
    // Member Variables
    private:
        StopWatchInterface *cudaTimer = nullptr;
        ImageUtils *imageUtils = nullptr;
        OtsuBinarizerCuda *binarizer = nullptr;

        uchar3 *hOldImage = nullptr, *hNewImage = nullptr, *hOutputImage = nullptr;
        uchar3 *dOldImage = NULL, *dNewImage = NULL, *dOutputImage = NULL;

    public:
        Logger *logger = nullptr;

    // Member Functions
    public:

        //* DEBUG Mode
        CudaChangeDetection(void);

        //* RELEASE Mode
        CudaChangeDetection(std::string logFilePath);

        ~CudaChangeDetection(void);

        /// @brief Change Detection Wrapper Function
        /// @param oldImagePath Image with old timestamp
        /// @param newImagePath Image with new timestamp
        /// @param outputPath Directory path to store output image
        /// @param grayscale  Output Image format : Grayscale (RED Color Changes) or Binary (WHITE Color Changes)
        /// @return Time required for the kernel to execute on GPU in seconds
        double detectChanges(std::string oldImagePath, std::string newImagePath, std::string outputPath, bool grayscale);

        
};

//* CUDA Kernel Declarations

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




