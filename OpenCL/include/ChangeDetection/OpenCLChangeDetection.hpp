#pragma once

#if (OS == 1)
    #include <windows.h>
#endif

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "../Common/Macros.hpp"
#include "../Common/Logger.hpp"

#ifndef _HELPER_TIMER_H_
    #define _HELPER_TIMER_H_
    #include "../Common/helper_timer.h"
#endif

#include "../CLFW/CLFW.hpp"

#include "ImageUtils.hpp"
#include "OtsuBinarizerOpenCL.hpp"


class OpenCLChangeDetection
{
    // Member Variables
    private:
        StopWatchInterface *oclTimer = nullptr;
        ImageUtils *imageUtils = nullptr;
        OtsuBinarizerOpenCL *binarizer = nullptr;
        CLFW *clfw = nullptr;

        cl_mem deviceOldImage = NULL, deviceNewImage = NULL, deviceOutputImage = NULL;
        

    public:
        Logger *logger = nullptr;

    // Member Functions
    public:

        //* DEBUG Mode
        OpenCLChangeDetection(void);

        //* RELEASE Mode
        OpenCLChangeDetection(std::string logFilePath);

        ~OpenCLChangeDetection();

        /// @brief Change Detection Wrapper Function
        /// @param oldImagePath Image with old timestamp
        /// @param newImagePath Image with new timestamp
        /// @param outputPath Directory path to store output image
        /// @param grayscale  Output Image format : Grayscale (RED Color Changes) or Binary (WHITE Color Changes)
        /// @return Time required for the kernel to execute on GPU in seconds
        double detectChanges(std::string oldImagePath, std::string newImagePath, std::string outputPath, bool grayscale);
};


