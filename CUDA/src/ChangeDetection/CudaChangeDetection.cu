#include "../../include/ChangeDetection/CudaChangeDetection.cuh"

// CUDA Kernel Definitions
// __global__ void grayscaleChangeDetection(uchar_t *oldImageData, uchar_t *newImageData, uchar_t *outputImageData, int threshold)
// {
//     // Code


// }

// Member Function Definitions

//* DEBUG Mode
CudaChangeDetection::CudaChangeDetection(void)
{
    // Code
    imageUtils = new ImageUtils();
    binarizer = new OtsuBinarizerCuda();

    sdkCreateTimer(&cudaTimer);
}

//* RELEASE Mode
CudaChangeDetection::CudaChangeDetection(std::string logFilePath)
{
    // Code
    logger = Logger::getInstance(logFilePath);
    imageUtils = new ImageUtils();
    binarizer = new OtsuBinarizerCuda();

    sdkCreateTimer(&cudaTimer);
}

double CudaChangeDetection::detectChanges(std::string oldImagePath, std::string newImagePath, std::string outputPath, bool grayscale)
{
    // Variable Declarations
    cv::String outputImagePath;

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

    std::string outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes" + std::filesystem::path(oldImagePath).extension().string());

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

    //* CUDA Kernel Configuration
    size_t size = (oldImage.size().height * oldImage.size().width) / 3;
    std::cout << std::endl << "Image Size : " << size << std::endl;

    // dim3 BLOCKS((size + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);

    sdkStartTimer(&cudaTimer);
    {   
        //* 2. Ostu Thresholding
        // int threshold1 = binarizer->getThreshold(&oldImage, multiThreading, threadCount);
        // int threshold2 = binarizer->getThreshold(&newImage, multiThreading, threadCount);
        // int meanThreshold = (threshold1 + threshold2) / 2;
    
        //* 3. Differencing
        // if (grayscale)
            // grayscaleChangeDetection<<<BLOCKS, THREADS_PER_BLOCK>>>(oldImage.data, newImage.data, outputImage.data, 90);
        // else
        //     binaryChangeDetection<<<>>>
    }
    sdkStopTimer(&cudaTimer);
    double result = sdkGetTimerValue(&cudaTimer) / 1000.0;
    result = std::round(result / 0.001) * 0.001;
    
    // Save Image
    imageUtils->saveImage(outputImagePath, &outputImage);

    outputImage.release();
    newImage.release();
    oldImage.release();

    return result;
}

CudaChangeDetection::~CudaChangeDetection(void)
{
    // Code
    sdkDeleteTimer(&cudaTimer);
    cudaTimer = nullptr;

    delete binarizer;
    binarizer = nullptr;
    
    delete imageUtils;
    imageUtils = nullptr;

    #if RELEASE
        logger->deleteInstance();
    #endif
}
