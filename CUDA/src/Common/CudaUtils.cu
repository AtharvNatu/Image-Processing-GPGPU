#include "../../include/Common/CudaUtils.cuh"

// Function Definitions
void cudaMemAlloc(void **devPtr, size_t size)
{
    // Code
    cudaError_t result = cudaMalloc(devPtr, size);
    if (result != cudaSuccess)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Allocate Memory On GPU : %s", cudaGetErrorString(result), " ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Allocate Memory On GPU : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
        #endif

        exit(CUDA_ERROR);
    }
}

void cudaMemCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    // Code
    cudaError_t result = cudaMemcpy(dst, src, count, kind);
    if (result != cudaSuccess)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Copy Memory On GPU : %s", cudaGetErrorString(result), " ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Copy Memory From : " << src << " To " << dst << " : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
        #endif

        exit(CUDA_ERROR);
    }
}

void cudaMemFree(void **devPtr)
{
    // Code
    if (*devPtr)
    {
        cudaFree(*devPtr);
        *devPtr = NULL;
    }
}

void convertImageToPixelArr(uchar_t *imageData, uchar3 *pixelArray, size_t size)
{
    // Code
    for (size_t i = 0; i < size; i++, imageData += 3)
    {
        pixelArray[i].x = imageData[2];
        pixelArray[i].y = imageData[1];
        pixelArray[i].z = imageData[0];
    }
}

void convertPixelArrToImage(uchar3 *pixelArray, uchar_t *imageData, size_t size)
{
    // Code
    for (size_t i = 0; i < size; i++, imageData += 3)
    {
        imageData[2] = pixelArray[i].x;
        imageData[1] = pixelArray[i].y;
        imageData[0] = pixelArray[i].z;
    }
}
