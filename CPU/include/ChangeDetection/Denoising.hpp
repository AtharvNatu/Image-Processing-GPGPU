#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "../Common/Macros.hpp"
#include "../Common/Logger.hpp"

class Denoising
{
    private:

        // Denoising Algorithms
        void __gaussianFilter(uchar_t *image, int imageWidth, int imageHeight);
        void __nonLocalMeansFilter(uchar_t *image, int imageWidth, int imageHeight);

    public:
        void getWindow(uchar_t* imageData, uchar_t* window, int row, int column, int width, int size);
        void subtractKernels(uchar_t* k1, uchar_t* k2, double* result, int size);
        double computeKernelNorm(double* kernel, int size);
        void getGaussianKernel(double* kernel, int size);

        void gaussianDenoising(cv::Mat *image);
        void nlmDenoising(cv::Mat *image);
        
};

