#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <omp.h>

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

    // Member Functions
    public:
        CPUChangeDetection(void);
        ~CPUChangeDetection(void);

        // Image Functions
        cv::Mat loadImage(string imagePath);
        void saveImage(string imagePath, cv::Mat image);

        void __changeDetectionKernel(cv::Mat* oldImage, cv::Mat* newImage, cv::Mat* outputImage, int threadCount);
        double detectChanges(string oldInputImage, string newInputImage, string outputPath);
};


