#include "../../include/ChangeDetection/ChangeDetection.hpp"

#define THRESHOLD 90

// Function Definitions
cv::Mat loadImage(string imagePath)
{
    // Code
    cv::Mat image = imread(cv::String(imagePath));
    if (!image.data)
    {
        cerr << endl << "Error : Failed To Load Image ... Exiting !!!" << endl;
        exit(OPENCV_ERROR);
    }
    return image;
}

void saveImage(string imagePath, cv::Mat image)
{
    // Code
    if (!cv::imwrite(cv::String(imagePath), image))
    {
        cerr << endl << "Error : Failed To Save Image ... Exiting !!!" << endl;
        exit(OPENCV_ERROR);
    }
}

int get_num_threads(void) 
{
    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    return num_threads;
}

void __changeDetection(cv::Mat* oldImage, cv::Mat* newImage, cv::Mat* outputImage)
{
    // Variable Declarations
    uchar_t oldGreyValue, newGreyValue, difference;

    // Code
    #pragma omp parallel for private(oldGreyValue, newGreyValue, difference) collapse(2)
    for (int i = 0; i < oldImage->rows; i++)
    {
        for (int j = 0; j < oldImage->cols; j++)
        {
            // Get RGB Vector for current pixel
            Vec3b oldIntensityVector = oldImage->at<Vec3b>(i, j);
            Vec3b newIntenstiyVector = newImage->at<Vec3b>(i, j);

            // Y = 0.299 * R + 0.587 * G + 0.114 * B
            oldGreyValue = static_cast<uchar_t>(
                (0.299 * oldIntensityVector[2]) + 
                (0.587 * oldIntensityVector[1]) + 
                (0.114 * oldIntensityVector[0])
            );

            newGreyValue = static_cast<uchar_t>(
                (0.299 * newIntenstiyVector[2]) + 
                (0.587 * newIntenstiyVector[1]) + 
                (0.114 * newIntenstiyVector[0])
            );

            difference = abs(oldGreyValue - newGreyValue);

            if (difference >= THRESHOLD)
            {
                // Vec3b => B G R
                outputImage->at<Vec3b>(i, j)[0] = 0;
                outputImage->at<Vec3b>(i, j)[1] = 0;
                outputImage->at<Vec3b>(i, j)[2] = 255;  
            }
            else
            {

                outputImage->at<Vec3b>(i, j)[0] = oldGreyValue;
                outputImage->at<Vec3b>(i, j)[1] = oldGreyValue;
                outputImage->at<Vec3b>(i, j)[2] = oldGreyValue;
            }
        }
    }
}

double cpuDetectChanges(string oldInputImage, string newInputImage, string outputPath)
{
    // Variable Declarations
    cv::String outputImagePath;

    // Code

    // Check Validity of Input Images
    if (!filesystem::exists(oldInputImage) || !filesystem::exists(newInputImage))
    {
        cerr << endl << "Error : Invalid Input Image ... Exiting !!!" << endl;
        exit(FILE_ERROR);
    }

    // Input and Output File
    filesystem::path oldFilePath = filesystem::path(oldInputImage).stem();
    filesystem::path newFilePath = filesystem::path(newInputImage).stem();

    string outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes" + filesystem::path(oldInputImage).extension().string());

    #if (OS == 1)
        outputImagePath = outputPath + ("\\" + outputFileName);
    #elif (OS == 2 || OS == 3)
        outputImagePath = outputPath + ("/" + outputFileName);
    #endif

    // Load Images
    cv::Mat oldImage = loadImage(oldInputImage);
    cv::Mat newImage = loadImage(newInputImage);

    // Empty Output Image => CV_8UC3 = 3-channel RGB Image
    cv::Mat outputImage(oldImage.rows, oldImage.cols, CV_8UC3, Scalar(0, 0, 0));

    // CPU Change Detection
    // clock_t start = getClockTime();
    // {
    //     __changeDetection(&oldImage, &newImage, &outputImage);
    // }
    // clock_t end = getClockTime();
    // double result = getExecutionTime(start, end);
    // cout << endl << "Clock Result = " << result << endl;

    StopWatchInterface *timer = NULL;

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    {
        __changeDetection(&oldImage, &newImage, &outputImage);
    }
    sdkStopTimer(&timer);
    double result = sdkGetTimerValue(&timer) / 1000.0;
    sdkDeleteTimer(&timer);
    timer = NULL;
    
    // Save Image
    saveImage(outputImagePath, outputImage);

    outputImage.release();
    newImage.release();
    oldImage.release();

    return result;
}

