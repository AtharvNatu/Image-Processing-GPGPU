#include "../../include/ChangeDetection/ImageUtils.hpp"

// Member Function Definitions
cv::Mat ImageUtils::loadImage(string imagePath)
{
    // Code
    cv::Mat image = cv::imread(cv::String(imagePath));
    if (!image.data)
    {
        #if RELEASE
            Logger *logger = new Logger();
            logger->printLog("Error : Failed To Load Image ... Exiting !!!");
            delete logger;
            logger = nullptr;
            exit(OPENCV_ERROR);
        #else
            cerr << endl << "Error : Failed To Load Image ... Exiting !!!" << endl;
            exit(OPENCV_ERROR);
        #endif 
    }

    return image;
}

void ImageUtils::saveImage(string imagePath, cv::Mat image)
{
    // Code
    if (!cv::imwrite(cv::String(imagePath), image))
    {
        #if RELEASE
            Logger *logger = new Logger();
            logger->printLog("Error : Failed To Save Image ... Exiting !!!");
            delete logger;
            logger = nullptr;
            exit(OPENCV_ERROR);
        #else
            cerr << endl << "Error : Failed To Save Image ... Exiting !!!" << endl;
            exit(OPENCV_ERROR);
        #endif 
    }
}