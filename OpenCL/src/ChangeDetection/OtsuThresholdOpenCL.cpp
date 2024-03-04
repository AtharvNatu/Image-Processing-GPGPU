#include "../../include/ChangeDetection/OtsuThresholdOpenCL.hpp"

const char *oclHistogram = 
    "__kernel void oclHistogramKernel(__global int *pixelData, __global int *histogram, int numPixels)" \
	"{" \
		"__local int localHistogram[256];" \

        "int localId = get_local_id(0);" \
        "int globalId = get_global_id(0);" \

        "for (int i = localId; i < 256; i += get_local_size(0))" \
        "{" \
            "localHistogram[i] = 0;" \
        "}" \

        "barrier(CLK_LOCAL_MEM_FENCE);" \

        "for (int i = globalId; i < numPixels; i+= get_global_size(0))" \
        "{" \
            "atomic_add(&localHistogram[pixelData[i]], 1);" \
        "}" \

        "barrier(CLK_LOCAL_MEM_FENCE);" \

        "for (int i = localId; i < 256; i += get_local_size(0))" \
        "{" \
            "atomic_add(&histogram[i], localHistogram[i]);" \
        "}" \
	"}";

const char *oclClassVariances = 
    "__kernel void oclComputeClassVariances(__global double *histogram, double allProbabilitySum, __global double *betweenClassVariances, int totalPixels)" \
	"{" \
        "int pixelID = get_global_id(0);" \

        "double firstClassProbability = 0, secondClassProbability = 0;" \
        "double firstClassMean = 0, secondClassMean = 0;" \
        "double firstProbabilitySum = 0;" \

        "for (int i = 0; i <= pixelID % 256; i++)" \
        "{" \
            "firstClassProbability = firstClassProbability + histogram[i];" \
            "firstProbabilitySum += i * histogram[i];" \
        "}" \

        "secondClassProbability = 1 - firstClassProbability;" \

        "firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;" \
        "secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;" \

        "betweenClassVariances[pixelID] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);" \

	"}";


OtsuThresholdOpenCL::OtsuThresholdOpenCL(void)
{
    sdkCreateTimer(&gpuTimer);
}

// Method Definitions
std::vector<double> OtsuThresholdOpenCL::computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, size_t *pixelCount, double *gpuTime)
{
    // Variable Declarations
    uchar_t pixelValue = 0;
    std::vector<double> histogram(MAX_PIXEL_VALUE, 0.0);
    std::vector<uchar_t> occurences(MAX_PIXEL_VALUE);
    std::vector<uchar_t> imageVector;

    // Code
    imageVector = imageUtils->getRawPixelData(inputImage);
    size_t totalPixels = imageVector.size();
    *pixelCount = totalPixels;
    
    sdkStartTimer(&gpuTimer);
    {
        #pragma omp parallel firstprivate(pixelValue) shared(totalPixels, histogram, imageVector) num_threads(12)
        {
            int segmentSize = MAX_PIXEL_VALUE / 12;

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
    sdkStopTimer(&gpuTimer);
    *gpuTime += sdkGetTimerValue(&gpuTimer);

    return histogram;
}

int OtsuThresholdOpenCL::computeThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, double *gpuTime)
{
    // Variable Declarations
    double allProbabilitySum = 0, maxVariance = 0;
    double *hostBetweenClassVariances = nullptr;
    cl_mem deviceHistogram = nullptr, deviceBetweenClassVariances = nullptr;
    int threshold = 0;
    size_t totalPixels = 0;
    size_t globalWorkSize = 256;
    size_t localWorkSize = 1;
    std::vector<double> hostHistogram;

    // Code
    // double *hostHistogram = computeHistogram(inputImage, imageUtils, clfw, &totalPixels, gpuTime);
    hostHistogram = computeHistogram(inputImage, imageUtils, clfw, &totalPixels, gpuTime);


    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        allProbabilitySum += i * hostHistogram[i];

    hostBetweenClassVariances = new double[MAX_PIXEL_VALUE];
    memset(hostBetweenClassVariances, 0, MAX_PIXEL_VALUE);

    std::cout << std::endl << "1" << std::endl;
    deviceHistogram = clfw->oclCreateBuffer(CL_MEM_READ_ONLY, sizeof(double) * MAX_PIXEL_VALUE);
    deviceBetweenClassVariances = clfw->oclCreateBuffer(CL_MEM_READ_WRITE, sizeof(double) * MAX_PIXEL_VALUE);

     std::cout << std::endl << "2" << std::endl;
    clfw->oclWriteBuffer(deviceHistogram, sizeof(double) * MAX_PIXEL_VALUE, hostHistogram.data());
    clfw->oclWriteBuffer(deviceBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE, hostBetweenClassVariances);

     std::cout << std::endl << "3" << std::endl;
    clfw->oclCreateProgram(oclClassVariances);

     std::cout << std::endl << "4" << std::endl;
	clfw->oclCreateKernel("oclComputeClassVariances", "bdbi", deviceHistogram, allProbabilitySum, deviceBetweenClassVariances, (int)totalPixels);

    *gpuTime += clfw->oclExecuteKernel(globalWorkSize, localWorkSize, 1);
     std::cout << std::endl << "5" << std::endl;
    clfw->oclReadBuffer(deviceBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE, hostBetweenClassVariances);

     std::cout << std::endl << "6" << std::endl;
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

    std::cout << std::endl << "7" << std::endl;
    clfw->oclReleaseBuffer(deviceBetweenClassVariances);
    clfw->oclReleaseBuffer(deviceHistogram);

    delete[] hostBetweenClassVariances;
    hostBetweenClassVariances = nullptr;

    //! REMOVE THIS IF HISTOGRAM IS NEEDED
    // delete[] hostHistogram;
    // hostHistogram = nullptr;

    std::cout << std::endl << "Threshold = " << threshold << std::endl;

    return 0;
}

OtsuThresholdOpenCL::~OtsuThresholdOpenCL()
{
    if (gpuTimer)
    {
        sdkDeleteTimer(&gpuTimer);
        gpuTimer = nullptr;
    }
}