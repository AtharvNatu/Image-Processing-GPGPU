#pragma once

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "../Common/Macros.hpp"

using namespace std;
using namespace cv;

// Function Declarations

// Image Functions
cv::Mat loadImage(string imagePath);
void saveImage(string imagePath, cv::Mat image);

double cpuDetectChanges(string oldInputImage, string newInputImage, string outputPath);
void __changeDetection__(Mat oldImage, Mat newImage);
