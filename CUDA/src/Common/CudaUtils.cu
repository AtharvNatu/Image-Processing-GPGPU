#include "../../include/Common/CudaUtils.cuh"

// Function Definitions
void CudaUtils::memAlloc(void **devPtr, size_t size)
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

void CudaUtils::memSet(void *devPtr, int value, size_t count)
{
    // Code
    cudaError_t result = cudaMemset(devPtr, value, count);
    if (result != cudaSuccess)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Initialize Memory On GPU : %s", cudaGetErrorString(result), " ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Initialize Memory On GPU : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
        #endif

        exit(CUDA_ERROR);
    }
}

void CudaUtils::memCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    // Code
    cudaError_t result = cudaMemcpy(dst, src, count, kind);
    if (result != cudaSuccess)
    {
        if (kind == cudaMemcpyHostToDevice)
        {
            #if RELEASE
                Logger *logger = Logger::getInstance("IPUG.log");
                logger->printLog("Error : Failed To Copy Memory From CPU To GPU : %s", cudaGetErrorString(result), " ... Exiting !!!");
                logger->deleteInstance();
            #else
                std::cerr << std::endl << "Error : Failed To Copy Memory From CPU To GPU : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
            #endif
        }
        else if (kind == cudaMemcpyDeviceToHost)
        {
            #if RELEASE
                Logger *logger = Logger::getInstance("IPUG.log");
                logger->printLog("Error : Failed To Copy Memory From GPU To CPU : %s", cudaGetErrorString(result), " ... Exiting !!!");
                logger->deleteInstance();
            #else
                std::cerr << std::endl << "Error : Failed To Copy Memory From GPU To CPU : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
            #endif
        }
       
        exit(CUDA_ERROR);
    }
}

void CudaUtils::memFree(void **devPtr)
{
    // Code
    if (*devPtr)
    {
        if (cudaFree(*devPtr) == cudaSuccess);
            *devPtr = nullptr;
    }
}

void CudaUtils::createEvent(cudaEvent_t *event)
{
    // Code
    cudaError_t result = cudaEventCreate(event);
    if (result != cudaSuccess)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Create Event : %s", cudaGetErrorString(result), " ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Create Event : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
        #endif

        exit(CUDA_ERROR);
    }
}

void CudaUtils::recordEvent(cudaEvent_t event, cudaStream_t stream)
{
    // Code
    cudaError_t result = cudaEventRecord(event, stream);
    if (result != cudaSuccess)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Record Event : %s", cudaGetErrorString(result), " ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Record Event : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
        #endif

        exit(CUDA_ERROR);
    }
}

void CudaUtils::syncEvent(cudaEvent_t event)
{
    // Code
    cudaError_t result = cudaEventSynchronize(event);
    if (result != cudaSuccess)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Synchronize Event : %s", cudaGetErrorString(result), " ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Synchronize Event : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
        #endif

        exit(CUDA_ERROR);
    }
}

void CudaUtils::getEventElapsedTime(double *ms, cudaEvent_t start, cudaEvent_t end)
{
    // Code
    float elapsedTime = 0.0F;
    cudaError_t result = cudaEventElapsedTime(&elapsedTime, start, end);
    if (result != cudaSuccess)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Get Event Elapsed Time : %s", cudaGetErrorString(result), " ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Get Event Elapsed Time : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
        #endif

        exit(CUDA_ERROR);
    }
    *ms += elapsedTime;
}

void CudaUtils::destroyEvent(cudaEvent_t event)
{
    // Code
    cudaError_t result = cudaEventDestroy(event);
    if (result != cudaSuccess)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Synchronize Event : %s", cudaGetErrorString(result), " ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Synchronize Event : " << cudaGetErrorString(result) << " ... Exiting !!!" << std::endl;
        #endif

        exit(CUDA_ERROR);
    }
}

void CudaUtils::convertImageToPixelArr(uchar_t *imageData, uchar3 *pixelArray, size_t size)
{
    // Code
    for (size_t i = 0; i < size; i++, imageData += 3)
    {
        pixelArray[i].x = imageData[2];
        pixelArray[i].y = imageData[1];
        pixelArray[i].z = imageData[0];
    }
}

void CudaUtils::convertPixelArrToImage(uchar3 *pixelArray, uchar_t *imageData, size_t size)
{
    // Code
    for (size_t i = 0; i < size; i++, imageData += 3)
    {
        imageData[2] = pixelArray[i].x;
        imageData[1] = pixelArray[i].y;
        imageData[0] = pixelArray[i].z;
    }
}
