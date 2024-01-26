#include "../include/ChangeDetection/ChangeDetection.hpp"

void loadImage(string path)
{
    Mat image = imread(cv::String(path));
    if (!image.data)
    {
        cerr << endl << "Failed To Load Image ... Exiting !!!" << endl;
        exit(EXIT_FAILURE);
    }

    String windowName = "macOS OpenCV C++";
    namedWindow(windowName);
    imshow(windowName, image);

    waitKey(0);

    destroyWindow(windowName);
}

