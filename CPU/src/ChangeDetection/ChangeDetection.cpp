#include "../../include/ChangeDetection/ChangeDetection.hpp"

// Member Function Definitions
CPUChangeDetection::CPUChangeDetection(void)
{
    // Code
    logger = Logger::getInstance("./logs/IPUG.log");
    imageUtils = new ImageUtils();
    denoiser = new Denoising();
    binarizer = new OtsuBinarizer();

    sdkCreateTimer(&cpuTimer);
}

void CPUChangeDetection::__changeDetectionKernel(cv::Mat* oldImage, cv::Mat* newImage, cv::Mat* outputImage, int threshold, bool multiThreading, int threadCount)
{
    // Variable Declarations
    uchar_t oldGreyValue, newGreyValue, difference;

    // Code
    if (multiThreading)
    {
        #pragma omp parallel for private(oldGreyValue, newGreyValue, difference) num_threads(threadCount) collapse(2)
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

                if (difference >= threshold)
                {
                    // Vec3b => B G R

                    // 255 255 255 => For Black and White Image
                    outputImage->at<cv::Vec3b>(i, j)[0] = 255;
                    outputImage->at<cv::Vec3b>(i, j)[1] = 255;
                    outputImage->at<cv::Vec3b>(i, j)[2] = 255;  
                }
                // else
                // {
                //     outputImage->at<Vec3b>(i, j)[0] = oldGreyValue;
                //     outputImage->at<Vec3b>(i, j)[1] = oldGreyValue;
                //     outputImage->at<Vec3b>(i, j)[2] = oldGreyValue;
                // }
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

                if (difference >= threshold)
                {
                    // Vec3b => B G R

                    // 255 255 255 => For Black and White Image
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
        }
    }
    
}

double CPUChangeDetection::detectChanges(std::string oldInputImage, std::string newInputImage, std::string outputPath, bool multiThreading, int threadCount)
{
    // Variable Declarations
    cv::String outputImagePath;

    // Code

    //* Check Validity of Input Images
    if (!std::filesystem::exists(oldInputImage) || !std::filesystem::exists(newInputImage))
    {
        #if RELEASE
            logger->printLog("Error : Invalid Input Image ... Exiting !!!");
            exit(FILE_ERROR);
        #else
            std::cerr << std::endl << "Error : Invalid Input Image ... Exiting !!!" << std::endl;
            exit(FILE_ERROR);
        #endif
    }

    // Input and Output File
    std::filesystem::path oldFilePath = std::filesystem::path(oldInputImage).stem();
    std::filesystem::path newFilePath = std::filesystem::path(newInputImage).stem();

    std::string outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes" + std::filesystem::path(oldInputImage).extension().string());

    #if (OS == 1)
        outputImagePath = outputPath + ("\\" + outputFileName);
    #elif (OS == 2 || OS == 3)
        outputImagePath = outputPath + ("/" + outputFileName);
    #endif

    // Load Images
    cv::Mat oldImage = imageUtils->loadImage(oldInputImage);
    cv::Mat newImage = imageUtils->loadImage(newInputImage);

    //* Empty Output Image => CV_8UC3 = 3-channel RGB Image
    cv::Mat outputImage(oldImage.rows, oldImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    sdkStartTimer(&cpuTimer);
    {   
        //* Image Denoising using Gaussian Blur
    
        //* Old Image
    

        //* New Image
        

        //* CPU Change Detection 
        __changeDetectionKernel(
            &oldImage, 
            &newImage, 
            &outputImage, 
            binarizer->getThreshold(&newImage), //* Otsu Thresholding
            multiThreading, 
            threadCount
        );
    }
    sdkStopTimer(&cpuTimer);
    double result = sdkGetTimerValue(&cpuTimer) / 1000.0;
    
    // Save Image
    imageUtils->saveImage(outputImagePath, &outputImage);

    outputImage.release();
    newImage.release();
    oldImage.release();

    return result;
}

CPUChangeDetection::~CPUChangeDetection(void)
{
    // Code
    sdkDeleteTimer(&cpuTimer);
    cpuTimer = nullptr;

    delete binarizer;
    binarizer = nullptr;

    delete denoiser;
    denoiser = nullptr;
    
    delete imageUtils;
    imageUtils = nullptr;

    logger->deleteInstance();
}
