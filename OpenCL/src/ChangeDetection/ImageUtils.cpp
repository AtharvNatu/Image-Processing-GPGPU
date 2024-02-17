#include "../../include/ChangeDetection/ImageUtils.hpp"

// Member Function Definitions
cv::Mat ImageUtils::loadImage(std::string imagePath)
{
    // Code
    cv::Mat image = cv::imread(cv::String(imagePath));
    if (!image.data)
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Load Image ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Load Image ... Exiting !!!" << std::endl;
        #endif 

        exit(OPENCV_ERROR);
    }

    return image;
}

void ImageUtils::saveImage(std::string imagePath, cv::Mat *image)
{
    // Code
    if (!cv::imwrite(cv::String(imagePath), *image))
    {
        #if RELEASE
            Logger *logger = Logger::getInstance("IPUG.log");
            logger->printLog("Error : Failed To Save Image ... Exiting !!!");
            logger->deleteInstance();
        #else
            std::cerr << std::endl << "Error : Failed To Save Image ... Exiting !!!" << std::endl;
        #endif 

        exit(OPENCV_ERROR);
    }
}

std::vector<uchar_t> ImageUtils::getRawPixelData(cv::Mat* image)
{
    // Code
    std::vector<uchar_t> imageVector;

    if (image->isContinuous())
        imageVector.assign((uchar_t*)image->datastart, (uchar_t*)image->dataend);
    else
    {
        for (int i = 0; i < image->rows; i++)
        {
            imageVector.insert(
                imageVector.end(), 
                image->ptr<uchar_t>(i), 
                image->ptr<uchar_t>(i) + image->cols
            );
        }
    }

    return imageVector;
}

cv::Mat ImageUtils::getQuadChannelImage(cv::Mat *rgbImage)
{
    // Variable Declarations
    cv::Mat rgbChannels[3];
    std::vector<cv::Mat> channels;

    // Code
    cv::Mat alphaImage(rgbImage->rows, rgbImage->cols, CV_8UC4);

    // Split 3-channel image into RGB channels
    cv::split(*rgbImage, rgbChannels);

    // Create an alpha channel
    cv::Mat alphaChannel(rgbImage->size(), CV_8UC1, cv::Scalar(255));

    // Merge RGB channels and add Alpha Channel
    channels.push_back(rgbChannels[0]);
    channels.push_back(rgbChannels[1]);
    channels.push_back(rgbChannels[2]);
    channels.push_back(alphaChannel);

    // Merge channels into 4-channel image
    cv::merge(channels, alphaImage);

    rgbImage->release();

    return alphaImage;

}

