#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "../Common/Macros.hpp"
#include "../Common/Logger.hpp"

using namespace std;

class Denoising
{
    private:
        float *kernel;

        // Denoising Algorithms
        void __gaussianBlurKernel(uchar_t *inputImage, uchar_t *outputImage, int imageWidth, int imageHeight, float *kernel);

    public:
        Denoising(void);
        ~Denoising(void);
        void gaussianBlur(cv::Mat *inputImage, cv::Mat *outputImage);
};

