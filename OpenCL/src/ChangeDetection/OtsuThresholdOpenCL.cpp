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

const char *oclClassVariance = 
    "__kernel void oclInterClassVariance(__global double *histogram, __global double *interClassVariance, double allProbabilitySum, int totalPixels)" \
	"{" \
        "const float epsilon = 1e-5f;" \
        "float backgroundSum = 0.0f, foregroundSum = 0.0f;" \
        "float backgroundWeight = 0.0f, foregroundWeight = 0.0f;" \
        "float maxVariance = 0.0f;" \
        "int threshold = 0;" \
        
        "int pixelID = get_global_id(0);" \

        "if (pixelID < 256)" \
        "{" \
            "for (int i = 0; i <= pixelID; ++i)" \
            "{" \
                "backgroundWeight += histogram[i];" \
                "if (backgroundWeight == 0)" \
                "{" \
                    "continue;" \
                "}" \

                "foregroundWeight = totalPixels - backgroundWeight;" \
                "if (foregroundWeight == 0)" \
                "{" \
                    "break;" \
                "}" \

                "backgroundSum += i * histogram[i];" \

                "float backgroundMean = backgroundSum / backgroundWeight;" \
                "float foregroundMean = (allProbabilitySum - backgroundSum) / foregroundWeight;" \

                "float variance = (float)backgroundWeight * (float)foregroundWeight * (backgroundMean - foregroundMean) * (backgroundMean - foregroundMean);" \

                "if (variance > maxVariance)" \
                "{" \
                    "maxVariance = variance;" \
                    "threshold = i;" \
                "}" \
            "}" \

            "if (get_global_id(0) == 0)" \
            "{" \
                "interClassVariance[0] = maxVariance / totalPixels;" // Normalized inter-class variance
                "interClassVariance[1] = (float)threshold;" // Threshold value
            "}" \

        "}" \

	"}";


OtsuThresholdOpenCL::OtsuThresholdOpenCL(void)
{
    sdkCreateTimer(&gpuTimer);
}

// Method Definitions
double* OtsuThresholdOpenCL::computeHistogram(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, size_t *pixelCount, double *gpuTime)
{
    // Variable Declarations
    cl_mem deviceHistogram = nullptr, deviceImage = nullptr;
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

    deviceHistogram = clfw->oclCreateBuffer(CL_MEM_READ_ONLY, histogramSize);
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

    clfw->oclReleaseBuffer(deviceHistogram);
    clfw->oclReleaseBuffer(deviceImage);
    hostHistogram.clear();

    return normalizedHistogram;
}

int OtsuThresholdOpenCL::computeThreshold(cv::Mat* inputImage, ImageUtils *imageUtils, CLFW *clfw, double *gpuTime)
{
    // Variable Declarations
    double allProbabilitySum = 0, maxVariance = 0;
    double *hostInterClassVariance = nullptr;
    cl_mem deviceHistogram = nullptr, deviceInterClassVariance = nullptr;
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

    hostInterClassVariance = new double[MAX_PIXEL_VALUE];
    memset(hostInterClassVariance, 0, MAX_PIXEL_VALUE);

    std::cout << std::endl << "hostInterClassVariance Done " << std::endl;

    deviceHistogram = clfw->oclCreateBuffer(CL_MEM_READ_ONLY, sizeof(double) * HIST_BINS);
    deviceInterClassVariance = clfw->oclCreateBuffer(CL_MEM_READ_WRITE, sizeof(double) * HIST_BINS);

    std::cout << std::endl << "oclCreateBuffer() Done " << std::endl;

    clfw->oclWriteBuffer(deviceHistogram, sizeof(double) * HIST_BINS, hostHistogram);
    clfw->oclWriteBuffer(deviceInterClassVariance, sizeof(double) * HIST_BINS, hostInterClassVariance);

    std::cout << std::endl << "oclWriteBuffer() Done " << std::endl;

    clfw->oclCreateProgram(oclClassVariance);

	clfw->oclCreateKernel("oclInterClassVariance", "bbdi", deviceHistogram, deviceInterClassVariance, allProbabilitySum, (int)totalPixels);

    *gpuTime += clfw->oclExecuteKernel(globalWorkSize, localWorkSize, 1);

    std::cout << std::endl << "oclExecuteKernel() Done " << std::endl;

    clfw->oclReadBuffer(deviceInterClassVariance, sizeof(double) * MAX_PIXEL_VALUE, hostInterClassVariance);

    std::cout << std::endl << "oclReadBuffer() Done " << std::endl;

    histFile = fopen("ocl-icv.txt", "wb");
        if (histFile == NULL)
            std::cerr << std::endl << "Failed to open file" << std::endl;
        for (int i = 0; i < HIST_BINS; i++)
            fprintf(histFile, "\tPixel value %d -> %.5f\n", i, hostInterClassVariance[i]);
        fprintf(histFile, "\n\n\n");
    fclose(histFile);

    // sdkStartTimer(&gpuTimer);
    // {
    //     for (int i = 0; i < MAX_PIXEL_VALUE; i++)
    //     {
    //         if (hostInterClassVariance[i] > maxVariance)
    //         {
    //             threshold = i;
    //             maxVariance = hostInterClassVariance[i];
    //         }
    //     }
    // }
    // sdkStopTimer(&gpuTimer);
    // *gpuTime += sdkGetTimerValue(&gpuTimer);

    clfw->oclReleaseBuffer(deviceInterClassVariance);
    clfw->oclReleaseBuffer(deviceHistogram);

    

    //! REMOVE THIS IF HISTOGRAM IS NEEDED
    // delete[] hostHistogram;
    // hostHistogram = nullptr;

    std::cout << std::endl << "Threshold = " << hostInterClassVariance[1] << std::endl;

    delete[] hostInterClassVariance;
    hostInterClassVariance = nullptr;

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