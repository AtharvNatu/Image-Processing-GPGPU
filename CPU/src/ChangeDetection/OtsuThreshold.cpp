#include "../../include/ChangeDetection/OtsuThreshold.hpp"

OtsuThresholdCPU::OtsuThresholdCPU(void)
{
    // Code
    sdkCreateTimer(&cpuTimer);
}

// Method Definitions
std::vector<double> OtsuThresholdCPU::computeHistogramST(cv::Mat* inputImage, ImageUtils *imageUtils, size_t* pixelCount, double *cpuTime)
{
    // Variable Declarations
    uchar_t pixelValue = 0;
    std::vector<double> histogram(MAX_PIXEL_VALUE);
    std::vector<uchar_t> occurences(MAX_PIXEL_VALUE);
    std::vector<uchar_t> imageVector;

    // Code
    imageVector = imageUtils->getRawPixelData(inputImage);
    size_t totalPixels = imageVector.size();
    *pixelCount = totalPixels;
    std::cout << std::endl << "Total Pixels = " << totalPixels << std::endl;
    
    sdkStartTimer(&cpuTimer);
    {
        for (size_t i = 0; i != totalPixels; i++)
        {
            pixelValue = imageVector[i];
            histogram[pixelValue]++;
        }

        //* Normalization
        for (int j = 0; j != MAX_PIXEL_VALUE; j++)
            histogram[j] = histogram[j] / totalPixels;
    }
    sdkStopTimer(&cpuTimer);
    *cpuTime += sdkGetTimerValue(&cpuTimer);

    return histogram;
}

std::vector<double> OtsuThresholdCPU::computeHistogramMT(cv::Mat* inputImage, ImageUtils *imageUtils, size_t* pixelCount, double *cpuTime, int threadCount)
{
    // Variable Declarations
    uchar_t pixelValue = 0;
    std::vector<double> histogram(MAX_PIXEL_VALUE);
    std::vector<uchar_t> occurences(MAX_PIXEL_VALUE);
    std::vector<uchar_t> imageVector;

    // Code
    imageVector = imageUtils->getRawPixelData(inputImage);
    size_t totalPixels = imageVector.size();
    *pixelCount = totalPixels;
    
    sdkStartTimer(&cpuTimer);
    {
        #pragma omp parallel firstprivate(pixelValue) shared(totalPixels, histogram, imageVector) num_threads(threadCount)
        {
            int segmentSize = MAX_PIXEL_VALUE / threadCount;

            #pragma omp for schedule(static, segmentSize)
            for (size_t i = 0; i < totalPixels; i++)
            {
                pixelValue = imageVector[i];
                #pragma omp atomic
                histogram[pixelValue]++;
            }

            #pragma omp barrier

            //* Normalization
            #pragma omp for schedule(static, segmentSize)
            for (int j = 0; j < MAX_PIXEL_VALUE; j++)
                histogram[j] = histogram[j] / totalPixels;
        }
    }
    sdkStopTimer(&cpuTimer);
    *cpuTime += sdkGetTimerValue(&cpuTimer);

    return histogram;
}

//* Single Threaded
int OtsuThresholdCPU::computeThresholdST(cv::Mat* inputImage, ImageUtils *imageUtils, double *cpuTime)
{
    // Variable Declarations
    int threshold = 0;
    double allProbabilitySum = 0;
    double backgroundProbability = 0, foregroundProbability = 0;
    double foregroundMean = 0, backgroundMean = 0;
    double interClassVariance, maxVariance = 0;
    double firstProbabilitySum = 0;
    size_t totalPixels = 0;

    // Code
    std::vector<double> histogram = computeHistogramST(inputImage, imageUtils, &totalPixels, cpuTime);

    FILE* histFile = fopen("icv-hist.txt", "wb");
    if (histFile == NULL)
            std::cerr << std::endl << "Failed to open file" << std::endl;

    sdkStartTimer(&cpuTimer);
    {
        for (int i = 0; i < MAX_PIXEL_VALUE; i++)
            allProbabilitySum += i * histogram[i];

        for (int j = 0; j < MAX_PIXEL_VALUE; j++)
        {
            backgroundProbability += histogram[j];
            foregroundProbability = 1 - backgroundProbability;
            firstProbabilitySum += j * histogram[j];

            foregroundMean = (double)firstProbabilitySum / (double)backgroundProbability;
            backgroundMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)foregroundProbability;

            interClassVariance = backgroundProbability * foregroundProbability * pow((foregroundMean - backgroundMean), 2);

            fprintf(histFile, "\tVariance value %d -> %.5f\n", j, interClassVariance);

            if (interClassVariance > maxVariance)
            {
                threshold = j;
                maxVariance = interClassVariance;
            }
        }
    }
    sdkStopTimer(&cpuTimer);
    *cpuTime += sdkGetTimerValue(&cpuTimer);

    fclose(histFile);

    std::cout << std::endl << "CPU Threshold ST = " << threshold << std::endl;
   
    return threshold;
}

//* Multi-threaded
int OtsuThresholdCPU::computeThresholdMT(cv::Mat* inputImage, ImageUtils *imageUtils, double *cpuTime, int threadCount)
{
    // Variable Declarations
    int threshold = 0;
    double allProbabilitySum = 0;
    size_t totalPixels = 0;

    // Code
    std::vector<double> histogram = computeHistogramMT(inputImage, imageUtils, &totalPixels, cpuTime, threadCount);

    sdkStartTimer(&cpuTimer);
    { 
        double* interClassVariances = new double[MAX_PIXEL_VALUE];

        #pragma omp parallel shared(allProbabilitySum, interClassVariances, totalPixels, histogram) num_threads(threadCount)
        {
            double backgroundProbability = 0, foregroundProbability = 0;
            double foregroundMean = 0, backgroundMean = 0, firstProbabilitySum = 0;

            int segmentSize = MAX_PIXEL_VALUE / threadCount;

            #pragma omp for schedule(static, segmentSize)
            for (int i = 0; i < MAX_PIXEL_VALUE; i++)
            {
                #pragma omp atomic
                allProbabilitySum += i * histogram[i];
                interClassVariances[i] = 0;
            }

            #pragma omp barrier

            #pragma omp for schedule(static, segmentSize)
            for (int j = 0; j < MAX_PIXEL_VALUE; j++)
            {
                backgroundProbability = 0;
                firstProbabilitySum = 0;

                for (int k = 0; k <= j % MAX_PIXEL_VALUE; k++)
                {
                    backgroundProbability = backgroundProbability + histogram[k];
                    firstProbabilitySum += k * histogram[k];
                }

                foregroundProbability = 1 - backgroundProbability;

                foregroundMean = (double)firstProbabilitySum / (double)backgroundProbability;
                backgroundMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)foregroundProbability;

                interClassVariances[j] = backgroundProbability * foregroundProbability * pow((foregroundMean - backgroundMean), 2);
            }

            #pragma omp barrier

            #pragma omp single
            {
                double maxVariance = 0;

                for (int l = 0; l < MAX_PIXEL_VALUE; l++)
                {
                    if (interClassVariances[l] > maxVariance)
                    {
                        threshold = l;
                        maxVariance = interClassVariances[l];
                    }
                }
            }
        }

        delete[] interClassVariances;
        interClassVariances = nullptr;
    }
    sdkStopTimer(&cpuTimer);
    *cpuTime += sdkGetTimerValue(&cpuTimer);

    std::cout << std::endl << "CPU Threshold OpenMP = " << threshold << std::endl;
   
    return threshold;
}

int OtsuThresholdCPU::getImageThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, bool multiThreading, int threadCount, double *cpuTime)
{
    // Variable Declrations
    int threshold = 0;

    // Code
    if (multiThreading)
        threshold = computeThresholdMT(inputImage, imageUtils, cpuTime, threadCount);
    else
        threshold = computeThresholdST(inputImage, imageUtils, cpuTime);
    
    return threshold;
}

OtsuThresholdCPU::~OtsuThresholdCPU()
{
    // Code
    if (cpuTimer)
    {
        sdkDeleteTimer(&cpuTimer);
        cpuTimer = nullptr;
    }
}
