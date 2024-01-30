#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "ImageUtils.hpp"
#include "Denoising.hpp"

#include "../Common/Macros.hpp"
#include "../Common/Logger.hpp"
#include "../Common/helper_timer.h"

using namespace std;
using namespace cv;

class CPUChangeDetection
{
    // Member Variables
    private:
        StopWatchInterface *cpuTimer = nullptr;
        Logger *logger = nullptr;
        ImageUtils *imageUtils = nullptr;
        Denoising *denoiser = nullptr;

        void __changeDetectionKernel(cv::Mat* oldImage, cv::Mat* newImage, cv::Mat* outputImage, int threadCount);

    // Member Functions
    public:
        CPUChangeDetection(void);
        ~CPUChangeDetection(void);
        double detectChanges(string oldInputImage, string newInputImage, string outputPath);
};


