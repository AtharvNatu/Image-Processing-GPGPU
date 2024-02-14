#include "../../include/ChangeDetection/OpenCLChangeDetection.hpp"

const char* oclChangeDetection =
	
	"__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;" \
	
	"__kernel void oclChangeDetectionKernel(__read_only image2d_t inputOld, __read_only image2d_t inputNew, __write_only image2d_t output, int threshold)" \
	"{" \
		"int2 gid = (int2)(get_global_id(0), get_global_id(1));" \
		
		"uint3 oldPixel, newPixel, finalPixelColor;" \
		"uint oldGrayVal, newGrayVal, difference;" \
		
		"int2 sizeOld = get_image_dim(inputOld);" \
		"int2 sizeNew = get_image_dim(inputNew);" \

		"if (all(gid < sizeOld) && all(gid < sizeNew))" \
		"{" \
			"oldPixel = read_imageui(inputOld, sampler, gid);" \
			"oldGrayVal = (0.3 * oldPixel.x) + (0.59 * oldPixel.y) + (0.11 * oldPixel.z);" \
			"newPixel = read_imageui(inputNew, sampler, gid);" \
			"newGrayVal = (0.3 * newPixel.x) + (0.59 * newPixel.y) + (0.11 * newPixel.z);" \
		"}" \

		"difference = abs_diff(oldGrayVal, newGrayVal);" \

		"if (difference >= threshold)" \
		"{" \
			"finalPixelColor = (uint3)(255, 0, 0);" \
		"}" \
		"else" \
		"{" \
			"finalPixelColor = (uint3)(oldGrayVal, oldGrayVal, oldGrayVal);" \
		"}" \
		
		"write_imageui(output, gid, finalPixelColor);" \
	"}";


// Member Function Definitions

//* DEBUG Mode
OpenCLChangeDetection::OpenCLChangeDetection(void)
{
    // Code
    imageUtils = new ImageUtils();
    binarizer = new OtsuBinarizerOpenCL();
    clfw = new CLFW();

    sdkCreateTimer(&oclTimer);
}

//* RELEASE Mode
OpenCLChangeDetection::OpenCLChangeDetection(std::string logFilePath)
{
    // Code
    logger = Logger::getInstance(logFilePath);
    imageUtils = new ImageUtils();
    binarizer = new OtsuBinarizerOpenCL();
    clfw = new CLFW();

    sdkCreateTimer(&oclTimer);
}

double OpenCLChangeDetection::detectChanges(std::string oldImagePath, std::string newImagePath, std::string outputPath, bool grayscale)
{
    // Variable Declarations
    cv::String outputImagePath;
    std::string outputFileName;
    double gpuTime = 0;

    // Code

    //* Check Validity of Input Images
    if (!std::filesystem::exists(oldImagePath) || !std::filesystem::exists(newImagePath))
    {
        #if RELEASE
            logger->printLog("Error : Invalid Input Image ... Exiting !!!");
        #else
            std::cerr << std::endl << "Error : Invalid Input Image ... Exiting !!!" << std::endl;
        #endif

        exit(FILE_ERROR);
    }

    // Input and Output File
    std::filesystem::path oldFilePath = std::filesystem::path(oldImagePath).stem();
    std::filesystem::path newFilePath = std::filesystem::path(newImagePath).stem();

    if (grayscale)
        outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes_Grayscale_OpenCL" + std::filesystem::path(oldImagePath).extension().string());
    else
        outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes_Binary_OpenCL" + std::filesystem::path(oldImagePath).extension().string());
    
    #if (OS == 1)
        outputImagePath = outputPath + ("\\" + outputFileName);
    #elif (OS == 2 || OS == 3)
        outputImagePath = outputPath + ("/" + outputFileName);
    #endif

    // Load Images
    cv::Mat oldImage = imageUtils->loadImage(oldImagePath);
    cv::Mat newImage = imageUtils->loadImage(newImagePath);

    //* 1. Preprocessing
    if (oldImage.cols != newImage.cols || oldImage.rows != newImage.rows)
    {
        #if RELEASE
            logger->printLog("Error : Invalid Spatial Resolution ... Input Images With Same Resolution ... Exiting !!!");     
        #else
            std::cerr << std::endl << "Error : Invalid Spatial Resolution ... Input Images With Same Resolution ... Exiting !!!" << std::endl;
        #endif

        newImage.release();
        oldImage.release();

        exit(FILE_ERROR);
    }

    std::cout << std::endl << "Image Channels = " << oldImage.channels() << std::endl;

    //* Empty Output Image => CV_8UC3 = 3-channel RGB Image
    cv::Mat outputImage(oldImage.rows, oldImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    //* 2. Ostu Thresholding
    clfw->initialize();

    // size_t pixelCount = 0;
    // binarizer->computeHistogram(&oldImage, imageUtils, clfw, &pixelCount, &gpuTime);
    // int threshold1 = binarizer->computeThreshold(&oldImage, imageUtils, multiThreading, threadCount, &cpuTime);
    // int threshold2 = binarizer->computeThreshold(&newImage, imageUtils, multiThreading, threadCount, &cpuTime);
    // int meanThreshold = (threshold1 + threshold2) / 2;

    //* 3. Differencing
    clfw->oclCreateImage(
        &deviceOldImage,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        oldImage.cols,
        oldImage.rows,
        oldImage.data
    );

    clfw->oclCreateImage(
        &deviceNewImage,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        newImage.cols,
        newImage.rows,
        newImage.data
    );

    clfw->oclCreateImage(
        &deviceOutputImage,
        CL_MEM_WRITE_ONLY,
        oldImage.cols,
        oldImage.rows,
        NULL
    );

    std::cout << std::endl << "1" << std::endl;
    clfw->oclCreateProgram(oclChangeDetection);

    clfw->oclCreateKernel("oclChangeDetectionKernel", "bbbi", deviceOldImage, deviceNewImage, deviceOutputImage, 90);
    std::cout << std::endl << "2" << std::endl;

    size_t globalSize[2] = 
    {
        static_cast<size_t>(oldImage.cols),
        static_cast<size_t>(oldImage.rows)
    };

    gpuTime += clfw->oclExecuteKernel(globalSize, 0, 2);

    std::cout << std::endl << "3" << std::endl;

    const size_t origin[3] = { 0, 0, 0 };
	const size_t region[3] = { static_cast<size_t>(oldImage.rows), static_cast<size_t>(oldImage.cols), 1 };

	// result = clEnqueueReadImage(oclCommandQueue, d_highlightedChanges, CL_TRUE, origin, region, 0, 0, highlightedChangesMat->data, 0, NULL, NULL);
	// if (result != CL_SUCCESS)
	// {
	// 	cerr << endl << "clEnqueueReadImage() Failed " << getErrorString(result) << "... Exiting !!!" << endl;
	// 	cleanup();
	// 	exit(EXIT_FAILURE);
	// }
    
	// cv::cvtColor(*highlightedChangesMat, *highlightedChangesMat, cv::COLOR_BGRA2RGBA);

    uchar_t* data = new uchar_t[oldImage.cols * oldImage.rows * 3];

    clfw->oclExecStatus(clEnqueueReadImage(clfw->oclCommandQueue, deviceOutputImage, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL));

    // clfw->oclReadImage(&deviceOutputImage, oldImage.cols, oldImage.rows, outputImage.data);
 
    outputImage.data = data;
    //    cv::cvtColor(outputImage, outputImage, cv::COLOR_BGRA2RGBA);

    delete[] data;
    data = nullptr;

    std::cout << std::endl << "4" << std::endl;

    // Save Image
    imageUtils->saveImage(outputImagePath, &outputImage);

    std::cout << std::endl << "5" << std::endl;

    outputImage.release();
    newImage.release();
    oldImage.release();

    clfw->oclReleaseBuffer(deviceOutputImage);
    clfw->oclReleaseBuffer(deviceNewImage);
    clfw->oclReleaseBuffer(deviceOldImage);

    clfw->uninitialize();

    return gpuTime;
}

OpenCLChangeDetection::~OpenCLChangeDetection(void)
{
    // Code
    sdkDeleteTimer(&oclTimer);
    oclTimer = nullptr;

    delete clfw;
    clfw = nullptr;

    delete binarizer;
    binarizer = nullptr;
    
    delete imageUtils;
    imageUtils = nullptr;

    #if RELEASE
        logger->deleteInstance();
    #endif
}
