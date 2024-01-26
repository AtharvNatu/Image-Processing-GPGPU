#include "../../include/ChangeDetection/ChangeDetection.hpp"

void cpuDetectChanges(string inputFile, string outputPath)
{
    Mat image = imread(cv::String(inputFile));
    if (!image.data)
    {
        cerr << endl << "Failed To Load Image ... Exiting !!!" << endl;
        exit(OPENCV_ERROR);
    }
}

