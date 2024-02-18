#include "../../include/ChangeDetection/OpenCLChangeDetection.hpp"

const char* oclChangeDetection =
	
	"__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;" \
	
	"__kernel void changeDetectionKernel(__read_only image2d_t inputOld, __read_only image2d_t inputNew, __write_only image2d_t output, int threshold, int grayscale)" \
	"{" \
		"int2 gid = (int2)(get_global_id(0), get_global_id(1));" \
		
		"uint4 oldPixel, newPixel, finalPixelColor;" \
		"uint oldGrayVal, newGrayVal, difference;" \
		
		"int2 sizeOld = get_image_dim(inputOld);" \
		"int2 sizeNew = get_image_dim(inputNew);" \

		"if (all(gid < sizeOld) && all(gid < sizeNew))" \
		"{" \
			"oldPixel = read_imageui(inputOld, sampler, gid);" \
			"oldGrayVal = (0.299 * oldPixel.x) + (0.587 * oldPixel.y) + (0.114 * oldPixel.z);" \
			"newPixel = read_imageui(inputNew, sampler, gid);" \
			"newGrayVal = (0.299 * newPixel.x) + (0.587 * newPixel.y) + (0.114 * newPixel.z);" \
		"}" \

		"difference = abs_diff(oldGrayVal, newGrayVal);" \

		"if (difference >= threshold)" \
		"{" \
            "if (grayscale)" \
            "{" \
			    "finalPixelColor = (uint4)(0, 0, 255, 255);" \
            "}" \
            "else" \
            "{" \
                "finalPixelColor = (uint4)(255, 255, 255, 255);" \
            "}" \
		"}" \
        "else" \
		"{" \
            "if (grayscale)" \
            "{" \
			    "finalPixelColor = (uint4)(oldGrayVal, oldGrayVal, oldGrayVal, 255);" \
            "}" \
            "else" \
            "{" \
                "finalPixelColor = (uint4)(0, 0, 0, 255);" \
            "}" \
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

    int width = oldImage.cols;
    int height = oldImage.rows;
    
    //* Convert to 32-bit images
    cv::Mat oldAlphaImage = imageUtils->getQuadChannelImage(&oldImage);
    cv::Mat newAlphaImage = imageUtils->getQuadChannelImage(&newImage);

    //* Empty Output Image
    cv::Mat outputImage(width, height, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    clfw->initialize();

    //* 2. Ostu Thresholding
    int threshold1 = binarizer->computeThreshold(&oldAlphaImage, imageUtils, clfw, &gpuTime);
    int threshold2 = binarizer->computeThreshold(&newAlphaImage, imageUtils, clfw, &gpuTime);
    int meanThreshold = (threshold1 + threshold2) / 2;

    //* 3. Differencing
    // clfw->oclCreateImage(&deviceOldImage, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width, height, oldAlphaImage.data);
    // clfw->oclCreateImage(&deviceNewImage, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width, height, newAlphaImage.data);
    // clfw->oclCreateImage(&deviceOutputImage, CL_MEM_WRITE_ONLY, width, height, NULL);

    // clfw->oclCreateProgram(oclChangeDetection);

    // if (grayscale)
    //     clfw->oclCreateKernel("changeDetectionKernel", "bbbii", deviceOldImage, deviceNewImage, deviceOutputImage, 90, 1);
    // else
    //     clfw->oclCreateKernel("changeDetectionKernel", "bbbii", deviceOldImage, deviceNewImage, deviceOutputImage, 90, 0);

    // size_t globalSize[2] = 
    // {
    //     static_cast<size_t>(width),
    //     static_cast<size_t>(height)
    // };
    // gpuTime += clfw->oclExecuteKernel(globalSize, 0, 2);

    // clfw->oclReadImage(&deviceOutputImage, width, height, outputImage.data);

    imageUtils->saveImage(outputImagePath, &outputImage);

    outputImage.release();
    newAlphaImage.release();
    oldAlphaImage.release();

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
