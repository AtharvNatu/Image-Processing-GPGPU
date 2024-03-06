#include "../../include/ChangeDetection/OtsuThresholdCuda.cuh"

OtsuThresholdCuda::OtsuThresholdCuda(void)
{
    // Code
    sdkCreateTimer(&gpuTimer);
}

//* CUDA Kernel Definitions
__global__ void cudaHistogram(uchar_t *pixelData, uint_t *histogram, int segmentSize, size_t totalPixels)
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
}

__global__ void cudaComputeClassVariances(double *histogram, double allProbabilitySum, double *betweenClassVariances, size_t totalPixels)
{
    // Code
    int pixelID = blockIdx.x * blockDim.x + threadIdx.x;

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
double* OtsuThresholdCuda::computeHistogram(cv::Mat* inputImage, size_t *pixelCount, double *gpuTime, ImageUtils *imageUtils, CudaUtils *cudaUtils)
{
    // Variable Declarations
    uint_t *hostHistogram = nullptr, *deviceHistogram = nullptr;
    uchar_t *devicePixelData = nullptr;
    double *normalizedHistogram = nullptr;

    // Code
    std::vector<uchar_t> imageData = imageUtils->getRawPixelData(inputImage);
    size_t totalPixels = imageData.size();
    *pixelCount = totalPixels;
    std::cout << std::endl << "Total Pixels = " << totalPixels << std::endl;

    hostHistogram = new uint_t[HIST_BINS];
    memset(hostHistogram, 0, HIST_BINS);
    
    cudaUtils->memAlloc((void**)&deviceHistogram, sizeof(uint_t) * HIST_BINS);
    cudaUtils->memSet(deviceHistogram, 0, HIST_BINS * sizeof(uint_t));
    
    cudaUtils->memAlloc((void**)&devicePixelData, sizeof(uchar_t) * totalPixels);

    cudaUtils->memCopy(deviceHistogram, hostHistogram, sizeof(uint_t) * HIST_BINS, cudaMemcpyHostToDevice);
    cudaUtils->memCopy(devicePixelData, imageData.data(), sizeof(uchar_t) * totalPixels, cudaMemcpyHostToDevice);

    dim3 BLOCKS(((totalPixels / 3) + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);

    long segmentSize = ceil(totalPixels / (THREADS_PER_BLOCK * BLOCKS.x)) + 1;

    std::cout << std::endl << "Segment Size = " << segmentSize << std::endl;

    //* CUDA Kernel Call
    cudaUtils->createEvent(&start);
    cudaUtils->createEvent(&end);
    cudaUtils->recordEvent(start, 0);
    {
        cudaHistogram<<<BLOCKS, THREADS_PER_BLOCK>>>(devicePixelData, deviceHistogram, segmentSize, totalPixels);
    }
    cudaUtils->recordEvent(end, 0);
    cudaUtils->syncEvent(end);
    cudaUtils->getEventElapsedTime(gpuTime, start, end);

    cudaUtils->memCopy(hostHistogram, deviceHistogram, sizeof(uint_t) * HIST_BINS, cudaMemcpyDeviceToHost);

    //* Normalize Host Histogram
    normalizedHistogram = new double[HIST_BINS];
    sdkStartTimer(&gpuTimer);
    {
        for (int i = 0; i < HIST_BINS; i++)
            normalizedHistogram[i] = (double)hostHistogram[i] / (double)totalPixels;
    }
    sdkStopTimer(&gpuTimer);
    *gpuTime += sdkGetTimerValue(&gpuTimer);

    cudaUtils->memFree((void**)&devicePixelData);
    cudaUtils->memFree((void**)&deviceHistogram);

    delete[] hostHistogram;
    hostHistogram = nullptr;

    return normalizedHistogram;
}

int OtsuThresholdCuda::computeThreshold(cv::Mat* inputImage, double *gpuTime, ImageUtils *imageUtils, CudaUtils *cudaUtils)
{
    // Variable Declarations
    double allProbabilitySum = 0, maxVariance = 0;
    double *hostBetweenClassVariances = nullptr, *deviceHistogram = nullptr, *deviceBetweenClassVariances = nullptr;
    int threshold = 0;
    size_t totalPixels = 0;

    // Code
    double *hostHistogram = computeHistogram(inputImage, &totalPixels, gpuTime, imageUtils, cudaUtils);

    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        allProbabilitySum += i * hostHistogram[i];

    hostBetweenClassVariances = new double[MAX_PIXEL_VALUE];
    memset(hostBetweenClassVariances, 0, MAX_PIXEL_VALUE);

    cudaUtils->memAlloc((void**)&deviceHistogram, sizeof(double) * MAX_PIXEL_VALUE);
    cudaUtils->memAlloc((void**)&deviceBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE);

    cudaUtils->memCopy(deviceHistogram, hostHistogram, sizeof(double) * MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);
    cudaUtils->memCopy(deviceBetweenClassVariances, hostBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);
    
    //* CUDA Kernel Configuration
    int NUM_BLOCKS = 1;
    int THREADS = THREADS_PER_BLOCK / 4;

    //* CUDA Kernel Call
    cudaUtils->recordEvent(start, 0);
    {
        cudaComputeClassVariances<<<NUM_BLOCKS, THREADS>>>(deviceHistogram, allProbabilitySum, deviceBetweenClassVariances, totalPixels);
    }
    cudaUtils->recordEvent(end, 0);
    cudaUtils->syncEvent(end);
    cudaUtils->getEventElapsedTime(gpuTime, start, end);

    cudaUtils->memCopy(hostBetweenClassVariances, deviceBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE, cudaMemcpyDeviceToHost);

    FILE* histFile = fopen("ocl-icv-hist.txt", "wb");
        if (histFile == NULL)
            std::cerr << std::endl << "Failed to open file" << std::endl;
        for (int i = 0; i < HIST_BINS; i++)
            fprintf(histFile, "\tVariance value %d -> %.5f\n", i, hostBetweenClassVariances[i]);
        fprintf(histFile, "\n\n\n");
    fclose(histFile);

    sdkStartTimer(&gpuTimer);
    {
        for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        {
            if (hostBetweenClassVariances[i] > maxVariance)
            {
                threshold = i;
                maxVariance = hostBetweenClassVariances[i];
            }
        }
    }
    sdkStopTimer(&gpuTimer);
    *gpuTime += sdkGetTimerValue(&gpuTimer);

    cudaUtils->destroyEvent(end);
    cudaUtils->destroyEvent(start);

    cudaUtils->memFree((void**)&deviceBetweenClassVariances);
    cudaUtils->memFree((void**)&deviceHistogram);

    delete[] hostBetweenClassVariances;
    hostBetweenClassVariances = nullptr;

    delete[] hostHistogram;
    hostHistogram = nullptr;

    std::cout << std::endl << "CUDA Threshold = " << threshold << std::endl;

    return threshold;
}   

OtsuThresholdCuda::~OtsuThresholdCuda()
{
    // Code
    if (gpuTimer)
    {
        sdkDeleteTimer(&gpuTimer);
        gpuTimer = nullptr;
    }
}

