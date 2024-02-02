#include "../../include/ChangeDetection/OtsuBinarizer.hpp"

// Method Definitions
vector<double> OtsuBinarizer::getHistogram(cv::Mat* inputImage)
{
    // Variable Declarations
    uchar_t pixelValue = 0;
    vector<double> histogram(MAX_PIXEL_VALUE);
    vector<uchar_t> occurences(MAX_PIXEL_VALUE);
    vector<uchar_t> imageVector;

    // Code
    if (inputImage->isContinuous())
        // imageVector.assign(
        //     inputImage->data, 
        //     inputImage->data + inputImage->total() * inputImage->channels()
        // );
        imageVector.assign((uchar_t*)inputImage->datastart, (uchar_t*)inputImage->dataend);
    else
    {
        for (int i = 0; i < inputImage->rows; i++)
        {
            imageVector.insert(
                imageVector.end(), 
                inputImage->ptr<uchar_t>(i), 
                inputImage->ptr<uchar_t>(i) + inputImage->cols
            );
        }
    }

    size_t totalPixels = imageVector.size();

    for (vector<uchar_t>::size_type i = 0; i != totalPixels; i++)
    {
        pixelValue = imageVector[i];
        histogram[pixelValue]++;
    }

    //* Normalization
    for (vector<uchar_t>::size_type j = 0; j != MAX_PIXEL_VALUE; j++)
        histogram[j] = histogram[j] / totalPixels;

    // double value = 0;
	// for (int i = 0; i < MAX_PIXEL_VALUE; i++) {
	// 	value = histogram[i];
	// 	printf("\tPixel value %d -> %.5f\n", i, value);
	// }

    return histogram;
}

int OtsuBinarizer::getThreshold(cv::Mat* inputImage)
{
    // Variable Declarations
    int threshold = 0;

    //* Probability, Mean, Variance
    double firstClassProbability = 0, secondClassProbability = 0;
    double firstClassMean = 0, secondClassMean = 0;
    double betweenClassVariance, maxVariance = 0;
    double allProbabilitySum = 0, firstProbabilitySum = 0;

    // Code
    vector<double> histogram = getHistogram(inputImage);

    for (int i = 0; i < MAX_PIXEL_VALUE; i++)
        allProbabilitySum += i * histogram[i];

    for (int j = 0; j < MAX_PIXEL_VALUE; j++)
    {
        firstClassProbability += histogram[j];
        secondClassProbability = 1 - firstClassProbability;
        firstProbabilitySum += j * histogram[j];

        firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
        secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

        betweenClassVariance = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);

        if (betweenClassVariance > maxVariance)
        {
            threshold = j;
            maxVariance = betweenClassVariance;
        }
    }

    cout << endl << "Threshold : " << threshold << endl;
   
    return threshold;
}

// void OtsuBinarizer::binarize(cv::Mat* inputImage)
// {
//     // Code
//     inputImage = inputImage;

//     int threshold = getThreshold(inputImage);

//     vector<uchar_t> imagePixels = imageUtils->getRawData(inputImage);

//     for (vector<uchar_t>::size_type i = 0; i != imageUtils->getTotalPixels(inputImage); i++)
//     {
//         if ((int)imagePixels[i] > threshold)
//             imagePixels[i] = 255;
//         else
//             imagePixels[i] = 0;
//     }

//     memcpy(inputImage->data, imagePixels.data(), imagePixels.size() * sizeof(uchar_t));
// }
