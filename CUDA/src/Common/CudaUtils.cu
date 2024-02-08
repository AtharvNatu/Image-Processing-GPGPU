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
