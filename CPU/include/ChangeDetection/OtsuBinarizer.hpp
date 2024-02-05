#pragma once

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <omp.h>

#include "../Common/Macros.hpp"

using namespace std;

class OtsuBinarizer
{
    public:
        std::vector<double> getHistogram(cv::Mat* inputImage);
        int getThreshold(cv::Mat* inputImage);
        // void binarize(cv::Mat* inputImage);
};

