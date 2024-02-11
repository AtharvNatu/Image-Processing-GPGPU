#include "../../include/ChangeDetection/OtsuBinarizerCuda.cuh"

OtsuBinarizerCuda::OtsuBinarizerCuda(void)
{
    imageUtils = new ImageUtils();
}

//* CUDA Kernel Definitions
__global__ void cudaHistogram(uchar_t *pixelData, uint_t *histogram, long segmentSize, long totalPixels)
{
    // Code
    size_t pixelID = blockIdx.x * blockDim.x + threadIdx.x;

    size_t start = pixelID * segmentSize;

    for (size_t i = start; i < (start + segmentSize); i++)
    {
        if (i < totalPixels)
        {
            int pixelValue = (int)pixelData[i];
            atomicAdd(&histogram[pixelValue], 1);
        }
    }

    // __syncthreads();
}

__global__ void cudaComputeClassVariances(double *histogram, double allProbabilitySum, long totalPixels, double *betweenClassVariances)
{
    // Code
    size_t pixelID = blockIdx.x * blockDim.x + threadIdx.x;

    double firstClassProbability = 0, secondClassProbability = 0;
    double firstClassMean = 0, secondClassMean = 0;
    double firstProbabilitySum = 0;

    for (int i = 0; i <= pixelID % MAX_PIXEL_VALUE; i++)
    {
        firstClassProbability = firstClassProbability + histogram[i];
        firstProbabilitySum += i * histogram[i];
    }

    secondClassProbability = 1 - firstClassProbability;

    firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
    secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

    betweenClassVariances[pixelID] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);

    // __syncthreads();

}

// Method Definitions
double* OtsuBinarizerCuda::computeHistogram(cv::Mat* inputImage, long *pixelCount)
{
    // Variable Declarations
    uint_t *hostHistogram = nullptr, *deviceHistogram = nullptr;
    uchar_t *devicePixelData = nullptr;
    double *normalizedHistogram = nullptr;

    // Code
    std::vector<uchar_t> imageData = imageUtils->getRawPixelData(inputImage);
    long totalPixels = imageData.size();
    *pixelCount = totalPixels;

    hostHistogram = new uint_t[MAX_PIXEL_VALUE];
    for (uint_t i = 0; i < MAX_PIXEL_VALUE; i++)
        hostHistogram[i] = 0;
    
    cudaMemAlloc((void**)&deviceHistogram, sizeof(uint_t) * MAX_PIXEL_VALUE);
    cudaMemAlloc((void**)&devicePixelData, sizeof(uchar_t) * totalPixels);

    cudaMemCopy(deviceHistogram, hostHistogram, sizeof(uint_t) * MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);
    cudaMemCopy(devicePixelData, imageData.data(), sizeof(uchar_t) * totalPixels, cudaMemcpyHostToDevice);

    dim3 BLOCKS(((totalPixels / 3) + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);

    long segmentSize = ceil(totalPixels / (THREADS_PER_BLOCK * BLOCKS.x)) + 1;
    std::cout << std::endl << "Segment Size = " << segmentSize << std::endl;

    cudaHistogram<<<BLOCKS, THREADS_PER_BLOCK>>>(devicePixelData, deviceHistogram, segmentSize, totalPixels);

    cudaMemCopy(hostHistogram, deviceHistogram, sizeof(uint_t) * MAX_PIXEL_VALUE, cudaMemcpyDeviceToHost);

    //* Normalize Host Histogram
    normalizedHistogram = new double[MAX_PIXEL_VALUE];
    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        normalizedHistogram[i] = (double)hostHistogram[i] / (double)totalPixels;

    cudaMemFree((void**)&devicePixelData);
    cudaMemFree((void**)&deviceHistogram);

    delete[] hostHistogram;
    hostHistogram = nullptr;

    return normalizedHistogram;
}

int OtsuBinarizerCuda::computeThreshold(cv::Mat* inputImage)
{
    // Variable Declarations
    double allProbabilitySum = 0, maxVariance = 0;
    long totalPixels = 0;
    double *hostBetweenClassVariances = nullptr, *deviceHistogram = nullptr, *deviceBetweenClassVariances = nullptr;
    int threshold = 0;

    // Code
    double *hostHistogram = computeHistogram(inputImage, &totalPixels);

    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        allProbabilitySum += i * hostHistogram[i];

    hostBetweenClassVariances = new double[MAX_PIXEL_VALUE];
    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        hostBetweenClassVariances[i] = 0;

    cudaMemAlloc((void**)&deviceHistogram, sizeof(double) * MAX_PIXEL_VALUE);
    cudaMemAlloc((void**)&deviceBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE);

    std::cout << std::endl << "Done 1" << std::endl;

    cudaMemCopy(deviceHistogram, hostHistogram, sizeof(double) * MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);
    cudaMemCopy(deviceBetweenClassVariances, hostBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);

     std::cout << std::endl << "Done 2" << std::endl;
    
    dim3 BLOCKS(((totalPixels / 3) + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);

    //* CUDA Kernel Call
    cudaComputeClassVariances<<<BLOCKS, THREADS_PER_BLOCK>>>(deviceHistogram, allProbabilitySum, totalPixels, deviceBetweenClassVariances);

    cudaMemCopy(hostBetweenClassVariances, deviceBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE, cudaMemcpyDeviceToHost);

    std::cout << std::endl << "Done 3" << std::endl;

    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
    {
        if (hostBetweenClassVariances[i] > maxVariance)
        {
            threshold = i;
            maxVariance = hostBetweenClassVariances[i];
        }
    }

    cudaMemFree((void**)&deviceBetweenClassVariances);
    cudaMemFree((void**)&deviceHistogram);

    delete[] hostBetweenClassVariances;
    hostBetweenClassVariances = nullptr;

    //! REMOVE THIS IF HISTOGRAM IS NEEDED
    delete[] hostHistogram;
    hostHistogram = nullptr;

    return threshold;
}   

OtsuBinarizerCuda::~OtsuBinarizerCuda(void)
{
    delete imageUtils;
    imageUtils = nullptr;
}

