#include "../../include/ChangeDetection/ChangeDetection.hpp"

// Member Function Definitions

//* DEBUG Mode
CPUChangeDetection::CPUChangeDetection(void)
{
    // Code
    imageUtils = new ImageUtils();
    otsuThreshold = new OtsuThresholdCPU();

    sdkCreateTimer(&cpuTimer);
}

//* RELEASE Mode
CPUChangeDetection::CPUChangeDetection(std::string logFilePath)
{
    // Code
    logger = Logger::getInstance(logFilePath);
    imageUtils = new ImageUtils();
    otsuThreshold = new OtsuThresholdCPU();

    sdkCreateTimer(&cpuTimer);
}

void CPUChangeDetection::__changeDetectionKernel(cv::Mat* oldImage, cv::Mat* newImage, cv::Mat* outputImage, bool grayscale, int threshold, bool multiThreading, int threadCount)
{
    // Variable Declarations
    uchar_t oldGreyValue, newGreyValue, difference;

    // Code
    if (multiThreading)
    {
        #pragma omp parallel for private(oldGreyValue, newGreyValue, difference) num_threads(threadCount)
        for (int i = 0; i < oldImage->rows; i++)
        {
            for (int j = 0; j < oldImage->cols; j++)
            {
                // Get RGB Vector for current pixel
                cv::Vec3b oldIntensityVector = oldImage->at<cv::Vec3b>(i, j);
                cv::Vec3b newIntensityVector = newImage->at<cv::Vec3b>(i, j);

                // Y = 0.299 * R + 0.587 * G + 0.114 * B
                oldGreyValue = static_cast<uchar_t>(
                    (0.299 * oldIntensityVector[2]) + 
                    (0.587 * oldIntensityVector[1]) + 
                    (0.114 * oldIntensityVector[0])
                );

                newGreyValue = static_cast<uchar_t>(
                    (0.299 * newIntensityVector[2]) + 
                    (0.587 * newIntensityVector[1]) + 
                    (0.114 * newIntensityVector[0])
                );

                difference = abs(oldGreyValue - newGreyValue);

                //* Grayscale Image with Changes marked in RED color
                if (grayscale)
                {
                    if (difference >= threshold)
                    {
                        outputImage->at<cv::Vec3b>(i, j)[0] = 0;
                        outputImage->at<cv::Vec3b>(i, j)[1] = 0;
                        outputImage->at<cv::Vec3b>(i, j)[2] = 255;
                    }
                    else
                    {
                        outputImage->at<cv::Vec3b>(i, j)[0] = oldGreyValue;
                        outputImage->at<cv::Vec3b>(i, j)[1] = oldGreyValue;
                        outputImage->at<cv::Vec3b>(i, j)[2] = oldGreyValue;
                    }
                }

                //* Binary Image with Changes marked in WHITE color
                else
                {
                    if (difference >= threshold)
                    {
                        outputImage->at<cv::Vec3b>(i, j)[0] = 255;
                        outputImage->at<cv::Vec3b>(i, j)[1] = 255;
                        outputImage->at<cv::Vec3b>(i, j)[2] = 255;  
                    }
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < oldImage->rows; i++)
        {
            for (int j = 0; j < oldImage->cols; j++)
            {
                // Get RGB Vector for current pixel
                cv::Vec3b oldIntensityVector = oldImage->at<cv::Vec3b>(i, j);
                cv::Vec3b newIntensityVector = newImage->at<cv::Vec3b>(i, j);

                // Y = 0.299 * R + 0.587 * G + 0.114 * B
                oldGreyValue = static_cast<uchar_t>(
                    (0.299 * oldIntensityVector[2]) + 
                    (0.587 * oldIntensityVector[1]) + 
                    (0.114 * oldIntensityVector[0])
                );

                newGreyValue = static_cast<uchar_t>(
                    (0.299 * newIntensityVector[2]) + 
                    (0.587 * newIntensityVector[1]) + 
                    (0.114 * newIntensityVector[0])
                );

                difference = abs(oldGreyValue - newGreyValue);

                //* Grayscale Image with Changes marked in RED color
                if (grayscale)
                {
                    if (difference >= threshold)
                    {
                        outputImage->at<cv::Vec3b>(i, j)[0] = 0;
                        outputImage->at<cv::Vec3b>(i, j)[1] = 0;
                        outputImage->at<cv::Vec3b>(i, j)[2] = 255;
                    }
                    else
                    {
                        outputImage->at<cv::Vec3b>(i, j)[0] = oldGreyValue;
                        outputImage->at<cv::Vec3b>(i, j)[1] = oldGreyValue;
                        outputImage->at<cv::Vec3b>(i, j)[2] = oldGreyValue;
                    }
                }

                //* Binary Image with Changes marked in WHITE color
                else
                {
                    if (difference >= threshold)
                    {
                        outputImage->at<cv::Vec3b>(i, j)[0] = 255;
                        outputImage->at<cv::Vec3b>(i, j)[1] = 255;
                        outputImage->at<cv::Vec3b>(i, j)[2] = 255;  
                    }
                        
                }
            }
        }
    }
}

double CPUChangeDetection::detectChanges(std::string oldImagePath, std::string newImagePath, std::string outputPath, bool grayscale, bool multiThreading, int threadCount)
{
    // Variable Declarations
    cv::String outputImagePath;
    std::string outputFileName;
    double cpuTime = 0;

    // Code

    //* Check Validity of Input Images
    if (!std::filesystem::exists(oldImagePath) || !std::filesystem::exists(newImagePath))
    {
        #if RELEASE
            logger->printLog("Error : Invalid Input Image ... Exiting !!!");
        #else
            std::cerr << std::endl << "Error : Invalid Input Image ... Exiting !!!" << std::endl;
        #endif

        exit(FILE_ERROR);
    }

    // Input and Output File
    std::filesystem::path oldFilePath = std::filesystem::path(oldImagePath).stem();
    std::filesystem::path newFilePath = std::filesystem::path(newImagePath).stem();

    if (grayscale)
        outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes_Grayscale_CPU" + std::filesystem::path(oldImagePath).extension().string());
    else
        outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes_Binary_CPU" + std::filesystem::path(oldImagePath).extension().string());
    
    #if (OS == 1)
        outputImagePath = outputPath + ("\\" + outputFileName);
    #elif (OS == 2 || OS == 3)
        outputImagePath = outputPath + ("/" + outputFileName);
    #endif

    // Load Images
    cv::Mat oldImage = imageUtils->loadImage(oldImagePath);
    cv::Mat newImage = imageUtils->loadImage(newImagePath);

    //* 1. Preprocessing
    if (oldImage.cols != newImage.cols || oldImage.rows != newImage.rows)
    {
        #if RELEASE
            logger->printLog("Error : Invalid Spatial Resolution ... Input Images With Same Resolution ... Exiting !!!");     
        #else
            std::cerr << std::endl << "Error : Invalid Spatial Resolution ... Input Images With Same Resolution ... Exiting !!!" << std::endl;
        #endif

        newImage.release();
        oldImage.release();

        exit(FILE_ERROR);
    }

    //* Empty Output Image => CV_8UC3 = 3-channel RGB Image
    cv::Mat outputImage(oldImage.rows, oldImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    //* 2. Ostu Thresholding
    int threshold1 = otsuThreshold->getImageThreshold(&oldImage, imageUtils, multiThreading, threadCount, &cpuTime);
    int threshold2 = otsuThreshold->getImageThreshold(&newImage, imageUtils, multiThreading, threadCount, &cpuTime);
    int meanThreshold = (threshold1 + threshold2) / 2;

    //* 3. Differencing
    sdkStartTimer(&cpuTimer);
    {   
        __changeDetectionKernel(
            &oldImage, 
            &newImage, 
            &outputImage,
            grayscale,
            meanThreshold,
            multiThreading, 
            threadCount
        );
    }
    sdkStopTimer(&cpuTimer);
    cpuTime += sdkGetTimerValue(&cpuTimer);

    //* Milliseconds to Seconds
    cpuTime /= 1000.0;

    //* Round to 3 Decimal Places
    cpuTime = std::round(cpuTime / 0.001) * 0.001;
    
    // Save Image
    imageUtils->saveImage(outputImagePath, &outputImage);

    outputImage.release();
    newImage.release();
    oldImage.release();

    return cpuTime;
}

CPUChangeDetection::~CPUChangeDetection(void)
{
    // Code
    sdkDeleteTimer(&cpuTimer);
    cpuTimer = nullptr;

    delete otsuThreshold;
    otsuThreshold = nullptr;
    
    delete imageUtils;
    imageUtils = nullptr;

    #if RELEASE
        logger->deleteInstance();
    #endif
}
