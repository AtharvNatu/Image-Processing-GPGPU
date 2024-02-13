#include "../../include/ChangeDetection/OtsuBinarizerOpenCL.hpp"

const char *oclHistogram = 
    "__kernel void oclHistogramKernel(__global uchar *pixelData, __global uint *histogram, int segmentSize, ulong totalPixels)" \
	"{" \
		"size_t pixelID = get_global_id(0);" \

        "size_t start = pixelID * segmentSize;" \
		
        "for (size_t i = start; i < (start + segmentSize); i++)" \
        "{" \
            "if (i < totalPixels)" \
            "{" \
                "int pixelValue = (int)pixelData[i];" \
                "atomic_add(&histogram[pixelValue], 1);" \
            "}"
        "}"
	"}";

OtsuBinarizerOpenCL::OtsuBinarizerOpenCL(void)
{
    sdkCreateTimer(&gpuTimer);
}

// Method Definitions
double* OtsuBinarizerOpenCL::computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, size_t *pixelCount, double *gpuTime)
{
    // Variable Declarations
    uint_t *hostHistogram = nullptr;
    double *normalizedHistogram = nullptr;
    cl_mem deviceHistogram = NULL;
    cl_mem devicePixelData = NULL;
    StopWatchInterface *gpuTimer = nullptr;

    // Code
    std::vector<uchar_t> imageData = imageUtils->getRawPixelData(inputImage);
    size_t totalPixels = imageData.size();
    *pixelCount = totalPixels;

    hostHistogram = new uint_t[MAX_PIXEL_VALUE];
    memset(hostHistogram, 0, MAX_PIXEL_VALUE);
    
    std::cout << std::endl << "1" << std::endl;
    deviceHistogram = clfw->oclCreateBuffer(CL_MEM_READ_WRITE, sizeof(uint_t) * MAX_PIXEL_VALUE);
    devicePixelData = clfw->oclCreateBuffer(CL_MEM_READ_ONLY, sizeof(uchar_t) * totalPixels);

    std::cout << std::endl << "2" << std::endl;
    clfw->oclCreateProgram(oclHistogram);

    std::cout << std::endl << "3" << std::endl;
	clfw->oclCreateKernel("oclHistogramKernel", "bbil", devicePixelData, deviceHistogram, 4, totalPixels);

    std::cout << std::endl << "4" << std::endl;
    clfw->oclWriteBuffer(deviceHistogram, sizeof(uint_t) * MAX_PIXEL_VALUE, hostHistogram);
    clfw->oclWriteBuffer(devicePixelData, sizeof(uchar_t) * totalPixels, imageData.data());

    std::cout << std::endl << "5" << std::endl;
    size_t globalSize[2] = 
    {
        (size_t)inputImage->size().width,
        (size_t)inputImage->size().height
    };
    *gpuTime += clfw->oclExecuteKernel(globalSize, 0, 2);
    std::cout << std::endl << "6" << std::endl;

    std::cout << std::endl << "7" << std::endl;
    clfw->oclReadBuffer(deviceHistogram, sizeof(uint_t) * MAX_PIXEL_VALUE, hostHistogram);
    
    std::cout << std::endl << "8" << std::endl;
    //* Normalize Host Histogram
    normalizedHistogram = new double[MAX_PIXEL_VALUE];
    sdkStartTimer(&gpuTimer);
    {
        for (int i = 0; i < MAX_PIXEL_VALUE; i++)
            normalizedHistogram[i] = (double)hostHistogram[i] / (double)totalPixels;
    }
    sdkStopTimer(&gpuTimer);
    *gpuTime += sdkGetTimerValue(&gpuTimer);

    std::cout << std::endl << "9" << std::endl;

    clfw->oclReleaseBuffer(devicePixelData);
    clfw->oclReleaseBuffer(deviceHistogram);

    delete[] hostHistogram;
    hostHistogram = nullptr;

    return normalizedHistogram;
}

int OtsuBinarizerOpenCL::computeThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, double *gpuTime)
{
    // Variable Declarations
    double allProbabilitySum = 0, maxVariance = 0;
    double *hostBetweenClassVariances = nullptr;
    cl_mem deviceHistogram = nullptr, deviceBetweenClassVariances = nullptr;
    int threshold = 0;
    size_t totalPixels = 0;

    // Code
    double *hostHistogram = computeHistogram(inputImage, imageUtils, clfw, &totalPixels, gpuTime);

    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        allProbabilitySum += i * hostHistogram[i];

    hostBetweenClassVariances = new double[MAX_PIXEL_VALUE];
    memset(hostBetweenClassVariances, 0, MAX_PIXEL_VALUE);

    deviceHistogram = clfw->oclCreateBuffer(CL_MEM_READ_ONLY, sizeof(double) * MAX_PIXEL_VALUE);
    deviceBetweenClassVariances = clfw->oclCreateBuffer(CL_MEM_READ_WRITE, sizeof(double) * MAX_PIXEL_VALUE);

    // clfw->oclCreateProgram(kernelSourceCode);

	// clfw->oclCreateKernel("vecAddGPU", "bbbi", deviceInput1, deviceInput2, deviceOutput, elements);

    clfw->oclWriteBuffer(deviceHistogram, sizeof(double) * MAX_PIXEL_VALUE, hostHistogram);
    clfw->oclWriteBuffer(deviceBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE, hostBetweenClassVariances);

    // *gpuTime += clfw->oclExecuteKernel(clfw->getGlobalWorkSize(localSize, elements), localSize, 1);

    clfw->oclReadBuffer(deviceBetweenClassVariances, sizeof(double) * MAX_PIXEL_VALUE, hostBetweenClassVariances);

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

    clfw->oclReleaseBuffer(deviceBetweenClassVariances);
    clfw->oclReleaseBuffer(deviceHistogram);

    delete[] hostBetweenClassVariances;
    hostBetweenClassVariances = nullptr;

    //! REMOVE THIS IF HISTOGRAM IS NEEDED
    delete[] hostHistogram;
    hostHistogram = nullptr;

    return threshold;
}

OtsuBinarizerOpenCL::~OtsuBinarizerOpenCL()
{
    if (gpuTimer)
    {
        sdkDeleteTimer(&gpuTimer);
        gpuTimer = nullptr;
    }
}