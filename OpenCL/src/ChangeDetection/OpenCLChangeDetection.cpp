#include "../../include/ChangeDetection/OpenCLChangeDetection.hpp"

// Member Function Definitions

//* DEBUG Mode
OpenCLChangeDetection::OpenCLChangeDetection(void)
{
    // Code
    imageUtils = new ImageUtils();
    binarizer = new OtsuBinarizerOpenCL();
    clfw = new CLFW();

    sdkCreateTimer(&oclTimer);
}

//* RELEASE Mode
OpenCLChangeDetection::OpenCLChangeDetection(std::string logFilePath)
{
    // Code
    logger = Logger::getInstance(logFilePath);
    imageUtils = new ImageUtils();
    binarizer = new OtsuBinarizerOpenCL();
    clfw = new CLFW();

    sdkCreateTimer(&oclTimer);
}

double OpenCLChangeDetection::detectChanges(std::string oldImagePath, std::string newImagePath, std::string outputPath, bool grayscale)
{
    // Variable Declarations
    cv::String outputImagePath;
    std::string outputFileName;
    double gpuTime = 0;

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
        outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes_Grayscale_OpenCL" + std::filesystem::path(oldImagePath).extension().string());
    else
        outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes_Binary_OpenCL" + std::filesystem::path(oldImagePath).extension().string());
    
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
    clfw->initialize();

    size_t pixelCount = 0;
    binarizer->computeHistogram(&oldImage, imageUtils, clfw, &pixelCount, &gpuTime);
    // int threshold1 = binarizer->computeThreshold(&oldImage, imageUtils, multiThreading, threadCount, &cpuTime);
    // int threshold2 = binarizer->computeThreshold(&newImage, imageUtils, multiThreading, threadCount, &cpuTime);
    // int meanThreshold = (threshold1 + threshold2) / 2;

    std::cout << std::endl << "Pixel Count = " << pixelCount << std::endl;

    //* 3. Differencing
    // sdkStartTimer(&cpuTimer);
    // {   
    //     __changeDetectionKernel(
    //         &oldImage, 
    //         &newImage, 
    //         &outputImage,
    //         grayscale,
    //         meanThreshold,
    //         multiThreading, 
    //         threadCount
    //     );
    // }
    // sdkStopTimer(&cpuTimer);
    // cpuTime += sdkGetTimerValue(&cpuTimer);

    // //* Milliseconds to Seconds
    // cpuTime /= 1000.0;

    // //* Round to 3 Decimal Places
    // cpuTime = std::round(cpuTime / 0.001) * 0.001;
    
    // Save Image
    imageUtils->saveImage(outputImagePath, &outputImage);

    outputImage.release();
    newImage.release();
    oldImage.release();

    clfw->uninitialize();

    return gpuTime;
}

OpenCLChangeDetection::~OpenCLChangeDetection(void)
{
    // Code
    sdkDeleteTimer(&oclTimer);
    oclTimer = nullptr;

    delete clfw;
    clfw = nullptr;

    delete binarizer;
    binarizer = nullptr;
    
    delete imageUtils;
    imageUtils = nullptr;

    #if RELEASE
        logger->deleteInstance();
    #endif
}
