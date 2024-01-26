#pragma once

#include <iostream>
#include <cstdlib>
#include <opencv4/opencv2/opencv.hpp>

#include "../Common/Macros.hpp"

using namespace std;
using namespace cv;

void cpuDetectChanges(string inputFile, string outputPath);
