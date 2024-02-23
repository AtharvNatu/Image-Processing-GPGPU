#include "../../include/ChangeDetection/OtsuBinarizerOpenCL.hpp"

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
            "printf(\"%d\", histogram[i]);"
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

OtsuBinarizerOpenCL::OtsuBinarizerOpenCL(void)
{
    sdkCreateTimer(&gpuTimer);
}

// Method Definitions
double* OtsuBinarizerOpenCL::computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, size_t *pixelCount, double *gpuTime)
{
    // Variable Declarations
    int *hostHistogram = nullptr;

    double *normalizedHistogram = nullptr;

    cl_mem deviceHistogram = NULL;
    cl_mem devicePixelData = NULL;

    int zero = 0;

    // Code
    std::vector<uchar_t> imageData = imageUtils->getRawPixelData(inputImage);
    // size_t totalPixels = imageData.size();
    // *pixelCount = totalPixels;

    // std::cout << std::endl << "Total Pixels = " << totalPixels << std::endl;
    // std::cout << std::endl << "Rows * Cols = " << inputImage->rows * inputImage->cols << std::endl;
    // std::cout << std::endl << "Rows * Cols * sizeof(int) = " << inputImage->rows * inputImage->cols * sizeof(int) << std::endl;

    const int histogramSize = HIST_BINS * sizeof(int);
    const int imageElements = inputImage->rows * inputImage->cols;
    const size_t imageSize = imageElements * sizeof(int);

    // hostHistogram = (int*)malloc(histogramSize * sizeof(int));
    // if (hostHistogram == NULL)
    //     std::cerr << std::endl << "Failed to allocate memory " << std::endl;
    hostHistogram = new int[HIST_BINS];
    
    
    devicePixelData = clfw->oclCreateBuffer(CL_MEM_READ_ONLY, imageSize);
    deviceHistogram = clfw->oclCreateBuffer(CL_MEM_WRITE_ONLY, histogramSize);

    clfw->oclWriteBuffer(devicePixelData, imageSize, imageData.data());
    
    clfw->oclFillBuffer(deviceHistogram, &zero, sizeof(int), 0, histogramSize);
    
    clfw->oclCreateProgram(oclHistogram);

	clfw->oclCreateKernel("oclHistogramKernel", "bbi", devicePixelData, deviceHistogram, imageElements);
    
    size_t localWorkSize = clfw->oclGetDeviceMaxWorkGroupSize();
    size_t globalWorkSize = (histogramSize % localWorkSize == 0) ? histogramSize : ((histogramSize / localWorkSize + 1) * localWorkSize);

    // size_t globalWorkSize[1];
    // globalWorkSize[0] = 1024;

    // size_t localWorkSize[1];
    // localWorkSize[0] = 64;

    std::cout << std::endl << "1" << std::endl;

    // *gpuTime += clfw->oclExecuteKernel(&globalWorkSize, &localWorkSize, 1);
    clfw->oclExecStatus(clEnqueueNDRangeKernel(
        clfw->oclCommandQueue,
        clfw->oclKernel,
        1,
        NULL,
        &globalWorkSize,
        &localWorkSize,
        0,
        NULL,
        NULL
    ));

    std::cout << std::endl << "2" << std::endl;

    // clfw->oclReadBuffer(deviceHistogram, histogramSize, hostHistogram);
    clfw->oclExecStatus(clEnqueueReadBuffer(clfw->oclCommandQueue, deviceHistogram, CL_TRUE, 0, histogramSize, hostHistogram, 0, NULL, NULL));

    std::cout << std::endl << "3" << std::endl;

    // std::cout << std::endl << "Before Normalization" << std::endl;
    // for (int i = 0; i < HIST_BINS; i++)
    //     std::cout << std::endl << hostHistogram[i];
    FILE* histFile = fopen("ocl-hist.txt", "wb");
    if (histFile == NULL)
        std::cerr << std::endl << "Failed to open file" << std::endl;
    fprintf(histFile, "Before Normalization\n\n");
    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        fprintf(histFile, "\tPixel value %d -> %.5d\n", i, hostHistogram[i]);
    fprintf(histFile, "\n\n\n");
    fclose(histFile);
    
    // //* Normalize Host Histogram
    // normalizedHistogram = new double[HIST_BINS];
    // sdkStartTimer(&gpuTimer);
    // {
    //     for (int i = 0; i < HIST_BINS; i++)
    //         normalizedHistogram[i] = (double)hostHistogram[i] / (double)totalPixels;
    // }
    // sdkStopTimer(&gpuTimer);
    // *gpuTime += sdkGetTimerValue(&gpuTimer);

    // std::cout << std::endl << "After Normalization" << std::endl;
    // for (int i = 0; i < HIST_BINS; i++)
    //     std::cout << std::endl << normalizedHistogram[i];

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
    size_t globalWorkSize = 256;
    size_t localWorkSize = 1;

    // Code
    double *hostHistogram = computeHistogram(inputImage, imageUtils, clfw, &totalPixels, gpuTime);

    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        allProbabilitySum += i * hostHistogram[i];

    hostBetweenClassVariances = new double[MAX_PIXEL_VALUE];
    memset(hostBetweenClassVariances, 0, MAX_PIXEL_VALUE);

    std::cout << std::endl << "1" << std::endl;
    deviceHistogram = clfw->oclCreateBuffer(CL_MEM_READ_ONLY, sizeof(double) * MAX_PIXEL_VALUE);
    deviceBetweenClassVariances = clfw->oclCreateBuffer(CL_MEM_READ_WRITE, sizeof(double) * MAX_PIXEL_VALUE);

     std::cout << std::endl << "2" << std::endl;
    clfw->oclWriteBuffer(deviceHistogram, sizeof(double) * MAX_PIXEL_VALUE, hostHistogram);
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
    delete[] hostHistogram;
    hostHistogram = nullptr;

    std::cout << std::endl << "Threshold = " << threshold << std::endl;

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