#include "../../include/ChangeDetection/ImageUtils.hpp"

// Member Function Definitions
cv::Mat ImageUtils::loadImage(string imagePath)
{
    // Code
    cv::Mat image = cv::imread(cv::String(imagePath));
    if (!image.data)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("./logs/IPUG.log");
            logger->printLog("Error : Failed To Load Image ... Exiting !!!");
            logger->deleteInstance();
            exit(OPENCV_ERROR);
        #else
            cerr << endl << "Error : Failed To Load Image ... Exiting !!!" << endl;
            exit(OPENCV_ERROR);
        #endif 
    }

    return image;
}

void ImageUtils::saveImage(string imagePath, cv::Mat *image)
{
    // Code
    if (!cv::imwrite(cv::String(imagePath), *image))
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("./logs/IPUG.log");
            logger->printLog("Error : Failed To Save Image ... Exiting !!!");
            logger->deleteInstance();
            exit(OPENCV_ERROR);
        #else
            cerr << endl << "Error : Failed To Save Image ... Exiting !!!" << endl;
            exit(OPENCV_ERROR);
        #endif 
    }
}

