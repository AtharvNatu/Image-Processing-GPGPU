#pragma once

#if (OS == 1)
    #include <windows.h>
#endif

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "../Common/Macros.hpp"
#include "../Common/Threading.hpp"
#include "../Common/Logger.hpp"
#include "../Common/helper_timer.h"

#include "ImageUtils.hpp"
#include "Denoising.hpp"
#include "OtsuBinarizer.hpp"


class CPUChangeDetection
{
    // Member Variables
    private:
        StopWatchInterface *cpuTimer = nullptr;
        ImageUtils *imageUtils = nullptr;
        Denoising *denoiser = nullptr;
        OtsuBinarizer *binarizer = nullptr;

        /// @brief CPU Change Detection Kernel
        /// @param oldImage Image with old timestamp
        /// @param newImage Image with new timestamp
        /// @param outputImage Empty output image
        /// @param grayscale Output Image format : Grayscale (RED Color Changes) or Binary (WHITE Color Changes)
        /// @param threshold Threshold generated using Otsu Binarization
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        void __changeDetectionKernel(cv::Mat* oldImage, cv::Mat* newImage, cv::Mat* outputImage, bool grayscale, int threshold, bool multiThreading, int threadCount);

    public:
        Logger *logger = nullptr;

    // Member Functions
    public:

        //* DEBUG Mode
        CPUChangeDetection(void);

        //* RELEASE Mode
        CPUChangeDetection(std::string logFilePath);

        ~CPUChangeDetection(void);

        /// @brief Change Detection Wrapper Function
        /// @param oldImagePath Image with old timestamp
        /// @param newImagePath Image with new timestamp
        /// @param outputPath Directory path to store output image
        /// @param grayscale  Output Image format : Grayscale (RED Color Changes) or Binary (WHITE Color Changes)
        /// @param multiThreading Single Threaded (false) or MultiThreaded (true)
        /// @param threadCount Thread Count calculated automatically as per CPU, if multiThreading = true
        /// @return Time required for the kernel to execute on CPU in seconds
        double detectChanges(std::string oldImagePath, std::string newImagePath, std::string outputPath, bool grayscale, bool multiThreading, int threadCount);
};


