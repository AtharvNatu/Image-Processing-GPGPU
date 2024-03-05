#include "../../include/ChangeDetection/OtsuThresholdOpenCL.hpp"

const char *oclHistogram = 
    "__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;" \
	
	"__kernel void oclHistogramKernel(__read_only image2d_t inputImage, __global uint *histogram)" \
	"{" \
		"int2 pixelId = (int2)(get_global_id(0), get_global_id(1));" \
		
		"int2 imageSize = get_image_dim(inputImage);" \

		"if (all(pixelId < imageSize))" \
		"{" \
			"uint4 pixelValue = read_imageui(inputImage, sampler, pixelId);" \

            "if (pixelValue.x < 256)" \
                "atom_inc(&histogram[pixelValue.x]);" \

            "if (pixelValue.y < 256)" \
                "atom_inc(&histogram[pixelValue.y]);" \

            "if (pixelValue.z < 256)" \
                "atom_inc(&histogram[pixelValue.z]);" \

            "if (pixelValue.w < 256)" \
                "atom_inc(&histogram[pixelValue.w]);" \
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
double* OtsuThresholdOpenCL::computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, size_t *pixelCount, double *gpuTime)
{
    // Variable Declarations
    std::vector<uint_t> hostHistogram(HIST_BINS, 0);
    double *normalizedHistogram = nullptr;

    // Code
    std::vector<uchar_t> imageData = imageUtils->getRawPixelData(inputImage);
    size_t totalPixels = imageData.size();
    *pixelCount = totalPixels;

    int histogramSize = sizeof(uint_t) * hostHistogram.size();

    clfw->oclCreateImage(
        &deviceImage, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
        inputImage->cols, 
        inputImage->rows, 
        inputImage->data
    );

    deviceHistogram = clfw->oclCreateBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, histogramSize);
    clfw->oclWriteBuffer(deviceHistogram, histogramSize, hostHistogram.data());

    clfw->oclCreateProgram(oclHistogram);

    clfw->oclCreateKernel("oclHistogramKernel", "bb", deviceImage, deviceHistogram);

    size_t globalSize[2] = 
    {
        static_cast<size_t>(inputImage->cols),
        static_cast<size_t>(inputImage->rows)
    };

    *gpuTime += clfw->oclExecuteKernel(globalSize, 0, 2);

    clfw->oclReadBuffer(deviceHistogram, histogramSize, hostHistogram.data());

    //* Normalize Host Histogram
    normalizedHistogram = new double[HIST_BINS];
    sdkStartTimer(&gpuTimer);
    {
        for (int i = 0; i < HIST_BINS; i++)
            normalizedHistogram[i] = (double)hostHistogram[i] / (double)totalPixels;
    }
    sdkStopTimer(&gpuTimer);
    *gpuTime += sdkGetTimerValue(&gpuTimer);

    hostHistogram.clear();

    return normalizedHistogram;
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

    // Code
    double *hostHistogram = computeHistogram(inputImage, imageUtils, clfw, &totalPixels, gpuTime);

    FILE* histFile = fopen("ocl-hist.txt", "wb");
        if (histFile == NULL)
            std::cerr << std::endl << "Failed to open file" << std::endl;
        for (int i = 0; i < HIST_BINS; i++)
            fprintf(histFile, "\tPixel value %d -> %.5f\n", i, hostHistogram[i]);
        fprintf(histFile, "\n\n\n");
    fclose(histFile);

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