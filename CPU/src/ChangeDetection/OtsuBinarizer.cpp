#include "../../include/ChangeDetection/OtsuBinarizer.hpp"

OtsuBinarizerCPU::OtsuBinarizerCPU(void)
{
    // Code
    sdkCreateTimer(&cpuTimer);
}

// Method Definitions
std::vector<double> OtsuBinarizerCPU::computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, bool multiThreading, int threadCount, size_t* pixelCount, double *cpuTime)
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
        if (multiThreading)
        {
            #pragma omp parallel firstprivate(pixelValue) shared(totalPixels, histogram, imageVector) num_threads(threadCount)
            {
                int segmentSize = MAX_PIXEL_VALUE / threadCount;

                #pragma omp for schedule(static, segmentSize)
                for (int i = 0; i < totalPixels; i++)
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
        else
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
    }
    sdkStopTimer(&cpuTimer);
    *cpuTime += sdkGetTimerValue(&cpuTimer);

    return histogram;
}

int OtsuBinarizerCPU::computeThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, bool multiThreading, int threadCount, double *cpuTime)
{
    // Variable Declarations
    int threshold = 0;
    double allProbabilitySum = 0;
    size_t totalPixels = 0;

    // Code
    std::vector<double> histogram = computeHistogram(inputImage, imageUtils, multiThreading, threadCount, &totalPixels, cpuTime);

    sdkStartTimer(&cpuTimer);
    {
        if (multiThreading)
        {   
            double* betweenClassVariances = new double[MAX_PIXEL_VALUE];

            #pragma omp parallel shared(allProbabilitySum, betweenClassVariances, totalPixels, histogram) num_threads(threadCount)
            {
                double firstClassProbability = 0, secondClassProbability = 0;
                double firstClassMean = 0, secondClassMean = 0, firstProbabilitySum = 0;

                int segmentSize = MAX_PIXEL_VALUE / threadCount;

                #pragma omp for schedule(static, segmentSize)
                for (int i = 0; i < MAX_PIXEL_VALUE; i++)
                {
                    #pragma omp atomic
                    allProbabilitySum += i * histogram[i];
                    betweenClassVariances[i] = 0;
                }

                #pragma omp barrier

                #pragma omp for schedule(static, segmentSize)
                for (int j = 0; j < MAX_PIXEL_VALUE; j++)
                {
                    firstClassProbability = 0;
                    firstProbabilitySum = 0;

                    for (int k = 0; k <= j % MAX_PIXEL_VALUE; k++)
                    {
                        firstClassProbability = firstClassProbability + histogram[k];
                        firstProbabilitySum += k * histogram[k];
                    }

                    secondClassProbability = 1 - firstClassProbability;

                    firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
                    secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

                    betweenClassVariances[j] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
                }

                #pragma omp barrier

                #pragma omp single
                {
                    double maxVariance = 0;

                    for (int l = 0; l < MAX_PIXEL_VALUE; l++)
                    {
                        if (betweenClassVariances[l] > maxVariance)
                        {
                            threshold = l;
                            maxVariance = betweenClassVariances[l];
                        }
                    }
                }
            }

            delete[] betweenClassVariances;
            betweenClassVariances = nullptr;
        }
        else
        {
            //* Single Threaded
            double firstClassProbability = 0, secondClassProbability = 0;
            double firstClassMean = 0, secondClassMean = 0;
            double betweenClassVariance, maxVariance = 0;
            double firstProbabilitySum = 0;

            for (int i = 0; i < MAX_PIXEL_VALUE; i++)
                allProbabilitySum += i * histogram[i];

            for (int j = 0; j < MAX_PIXEL_VALUE; j++)
            {
                firstClassProbability = firstClassProbability + histogram[j];
                secondClassProbability = 1 - firstClassProbability;
                firstProbabilitySum = firstProbabilitySum + j * histogram[j];

                firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
                secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

                betweenClassVariance = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);

                if (betweenClassVariance > maxVariance)
                {
                    threshold = j;
                    maxVariance = betweenClassVariance;
                }
            }
        }
    }
    sdkStopTimer(&cpuTimer);
    *cpuTime += sdkGetTimerValue(&cpuTimer);
   
    return threshold;
}

OtsuBinarizerCPU::~OtsuBinarizerCPU()
{
    // Code
    if (cpuTimer)
    {
        sdkDeleteTimer(&cpuTimer);
        cpuTimer = nullptr;
    }
}
