#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "ImageUtils.hpp"
#include "Denoising.hpp"
#include "OtsuBinarizer.hpp"

#include "../Common/Threading.hpp"
#include "../Common/Macros.hpp"
#include "../Common/Logger.hpp"
#include "../Common/helper_timer.h"


class CPUChangeDetection
{
    // Member Variables
    private:
        StopWatchInterface *cpuTimer = nullptr;
        ImageUtils *imageUtils = nullptr;
        Denoising *denoiser = nullptr;
        OtsuBinarizer *binarizer = nullptr;

        void __changeDetectionKernel(cv::Mat* oldImage, cv::Mat* newImage, cv::Mat* outputImage, int threshold, bool multiThreading, int threadCount);

    public:
        Logger *logger = nullptr;

    // Member Functions
    public:
        CPUChangeDetection(void);
        ~CPUChangeDetection(void);
        double detectChanges(std::string oldInputImage, std::string newInputImage, std::string outputPath, bool multiThreading, int threadCount);
};


