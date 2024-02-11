#include "../../include/ChangeDetection/CudaChangeDetection.cuh"

// CUDA Kernel Definitions
__global__ void grayscaleChangeDetection(uchar3 *oldImage, uchar3 *newImage, uchar3 *outputImage, int threshold, size_t size)
{
    // Variable Declarations
    uchar_t oldGreyValue, newGreyValue, difference;

    // Code
    long pixelId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pixelId < size)
    {
        oldGreyValue = (uchar_t)(
                        (0.299 * (uchar_t)oldImage[pixelId].x) +
                        (0.587 * (uchar_t)oldImage[pixelId].y) +
                        (0.114 * (uchar_t)oldImage[pixelId].z)
                    );

        newGreyValue = (uchar_t)(
                        (0.299 * (uchar_t)newImage[pixelId].x) +
                        (0.587 * (uchar_t)newImage[pixelId].y) +
                        (0.114 * (uchar_t)newImage[pixelId].z)
                    );

        difference = abs(oldGreyValue - newGreyValue);

        if (difference >= threshold)
        {
            outputImage[pixelId].x = 255;
            outputImage[pixelId].y = 0;
            outputImage[pixelId].z = 0;
        }
        else
        {
            outputImage[pixelId].x = oldGreyValue;
            outputImage[pixelId].y = oldGreyValue;
            outputImage[pixelId].z = oldGreyValue;
        }
    }
}

__global__ void binaryChangeDetection(uchar3 *oldImage, uchar3 *newImage, uchar3 *outputImage, int threshold, size_t size)
{
    // Variable Declarations
    uchar_t oldGreyValue, newGreyValue, difference;

    // Code
    long pixelId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pixelId < size)
    {
        oldGreyValue = (uchar_t)(
                        (0.299 * (uchar_t)oldImage[pixelId].x) +
                        (0.587 * (uchar_t)oldImage[pixelId].y) +
                        (0.114 * (uchar_t)oldImage[pixelId].z)
                    );

        newGreyValue = (uchar_t)(
                        (0.299 * (uchar_t)newImage[pixelId].x) +
                        (0.587 * (uchar_t)newImage[pixelId].y) +
                        (0.114 * (uchar_t)newImage[pixelId].z)
                    );

        difference = abs(oldGreyValue - newGreyValue);

        if (difference >= threshold)
        {
            outputImage[pixelId].x = 255;
            outputImage[pixelId].y = 255;
            outputImage[pixelId].z = 255;
        }
    }
}

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
        outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes_Grayscale_CUDA" + std::filesystem::path(oldImagePath).extension().string());
    else
        outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes_Binary_CUDA" + std::filesystem::path(oldImagePath).extension().string());
    
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

    size_t size = oldImage.size().height * oldImage.size().width;

    //* 2. Ostu Thresholding
    int threshold1 = binarizer->computeThreshold(&oldImage, imageUtils, &gpuTime);
    int threshold2 = binarizer->computeThreshold(&newImage, imageUtils, &gpuTime);
    int meanThreshold = (threshold1 + threshold2) / 2;

    hostOldImage = new uchar3[size];
    hostNewImage = new uchar3[size];
    hostOutputImage = new uchar3[size];

    convertImageToPixelArr(oldImage.data, hostOldImage, size);
    convertImageToPixelArr(newImage.data, hostNewImage, size);
    
    cudaMemAlloc((void**)&deviceOldImage, size * sizeof(uchar3));
    cudaMemAlloc((void**)&deviceNewImage, size * sizeof(uchar3));
    cudaMemAlloc((void**)&deviceOutputImage, size * sizeof(uchar3));

    cudaMemCopy(deviceOldImage, hostOldImage, size * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemCopy(deviceNewImage, hostNewImage, size * sizeof(uchar3), cudaMemcpyHostToDevice);

    //* CUDA Kernel Configuration
    dim3 BLOCKS((size + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);

    //* 3. Differencing
    sdkStartTimer(&cudaTimer);
    {   
        if (grayscale)
            grayscaleChangeDetection<<<BLOCKS, THREADS_PER_BLOCK>>>(deviceOldImage, deviceNewImage, deviceOutputImage, meanThreshold, size);
        else
            binaryChangeDetection<<<BLOCKS, THREADS_PER_BLOCK>>>(deviceOldImage, deviceNewImage, deviceOutputImage, meanThreshold, size);
    }
    sdkStopTimer(&cudaTimer);
    gpuTime += sdkGetTimerValue(&cudaTimer);
    gpuTime /= 1000.0;  //* Milliseconds to Seconds

    cudaMemCopy(hostOutputImage, deviceOutputImage, size * sizeof(uchar3), cudaMemcpyDeviceToHost);
    convertPixelArrToImage(hostOutputImage, outputImage.data, size);
    
    //* Save Image
    imageUtils->saveImage(outputImagePath, &outputImage);

    cleanup();

    outputImage.release();
    newImage.release();
    oldImage.release();

    return gpuTime;
}

void CudaChangeDetection::cleanup(void)
{
    //* Cleanup Code
    cudaMemFree((void**)&deviceOutputImage);
    cudaMemFree((void**)&deviceNewImage);
    cudaMemFree((void**)&deviceOldImage);

    delete[] hostOutputImage;
    hostOutputImage = nullptr;

    delete[] hostNewImage;
    hostNewImage = nullptr;

    delete[] hostOldImage;
    hostOldImage = nullptr;
}

CudaChangeDetection::~CudaChangeDetection()
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
