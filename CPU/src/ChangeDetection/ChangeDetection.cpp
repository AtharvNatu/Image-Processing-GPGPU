#include "../../include/ChangeDetection/ChangeDetection.hpp"

// Function Definitions
cv::Mat loadImage(string imagePath)
{
    // Code
    cv::Mat image = imread(cv::String(imagePath));
    if (!image.data)
    {
        cerr << endl << "Error : Failed To Load Image ... Exiting !!!" << endl;
        exit(OPENCV_ERROR);
    }
    return image;
}

void saveImage(string imagePath, cv::Mat image)
{
    // Code
    if (!cv::imwrite(cv::String(imagePath), image))
    {
        cerr << endl << "Error : Failed To Save Image ... Exiting !!!" << endl;
        exit(OPENCV_ERROR);
    }
}

double cpuDetectChanges(string oldInputImage, string newInputImage, string outputPath)
{
    // Variable Declarations
    cv::String outputImagePath;

    // Code

    // Check Validity of Input Images
    if (!filesystem::exists(oldInputImage) || !filesystem::exists(newInputImage))
    {
        cerr << endl << "Error : Invalid Input Image ... Exiting !!!" << endl;
        exit(FILE_ERROR);
    }

    // Input and Output File
    filesystem::path oldFilePath = filesystem::path(oldInputImage).stem();
    filesystem::path newFilePath = filesystem::path(newInputImage).stem();

    string outputFileName = oldFilePath.string() + ("_" + newFilePath.string()) + ("_Changes" + filesystem::path(oldInputImage).extension().string());

    #if (OS == 1)
        outputImagePath = outputPath + ("\\" + outputFileName);
    #elif (OS == 2 || OS == 3)
        outputImagePath = outputPath + ("/" + outputFileName);
    #endif

    // Load Images
    cv::Mat oldImage = loadImage(oldInputImage);
    cv::Mat newImage = loadImage(newInputImage);

    cv::Mat outputImage = newImage.clone();

    // Save Image
    saveImage(outputImagePath, outputImage);

    outputImage.release();
    newImage.release();
    oldImage.release();

    return 0;
}

