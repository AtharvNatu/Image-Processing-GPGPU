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

    __syncthreads();
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
}

// Method Definitions
double* OtsuBinarizerCuda::computeHistogram(cv::Mat* inputImage, long *pixelCount)
{
    // Code
    std::vector<uchar_t> imageData = imageUtils->getRawPixelData(inputImage);
    long totalPixels = imageData.size();
    *pixelCount = totalPixels;

    hostHistogram = new uint_t[MAX_PIXEL_VALUE];
    normalizedHistogram = new double[MAX_PIXEL_VALUE];

    memset(hostHistogram, 0, MAX_PIXEL_VALUE);

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
    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        normalizedHistogram[i] = (double)hostHistogram[i] / (double)totalPixels;

    cudaMemFree((void**)&devicePixelData);
    cudaMemFree((void**)&deviceHistogram);

    delete[] hostHistogram;
    hostHistogram = nullptr;

    return normalizedHistogram;
}

OtsuBinarizerCuda::~OtsuBinarizerCuda(void)
{
    delete imageUtils;
    imageUtils = nullptr;
}

