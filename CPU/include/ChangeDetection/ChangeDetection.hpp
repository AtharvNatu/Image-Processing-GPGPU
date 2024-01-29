#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "../Common/Macros.hpp"
#include "../Common/Timer.hpp"
#include "../Common/helper_timer.h"

using namespace std;
using namespace cv;

// Function Declarations

// Image Functions
cv::Mat loadImage(string imagePath);
void saveImage(string imagePath, cv::Mat image);

void __changeDetection(cv::Mat* oldImage, cv::Mat* newImage, cv::Mat* outputImage);
double cpuDetectChanges(string oldInputImage, string newInputImage, string outputPath);

