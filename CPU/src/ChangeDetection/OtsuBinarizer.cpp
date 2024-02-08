#include "../../include/ChangeDetection/OtsuBinarizer.hpp"

// Method Definitions
std::vector<double> OtsuBinarizer::getHistogram(cv::Mat* inputImage, bool multiThreading, int threadCount, size_t* pixelCount)
{
    // Variable Declarations
    uchar_t pixelValue = 0;
    std::vector<double> histogram(MAX_PIXEL_VALUE);
    std::vector<uchar_t> occurences(MAX_PIXEL_VALUE);
    std::vector<uchar_t> imageVector;

    // Code
    if (inputImage->isContinuous())
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
    *pixelCount = totalPixels;
    
    if (multiThreading)
    {
        #pragma omp parallel firstprivate(pixelValue) shared(totalPixels, histogram, imageVector) num_threads(threadCount)
        {
            int segmentSize = MAX_PIXEL_VALUE / threadCount;

            #pragma omp for schedule(static, segmentSize)
            for (size_t i = 0; i < totalPixels; i++)
            {
                pixelValue = imageVector[i];
                #pragma omp atomic
                histogram[pixelValue]++;
            }

            #pragma omp barrier

            //* Normalization
            #pragma omp for schedule(static, segmentSize)
            for (int j = 0; j < MAX_PIXEL_VALUE; j++)
                histogram[j] = histogram[j] / totalPixels;

        }
    }
    else
    {
        for (size_t i = 0; i != totalPixels; i++)
        {
            pixelValue = imageVector[i];
            histogram[pixelValue]++;
        }

        //* Normalization
        for (int j = 0; j != MAX_PIXEL_VALUE; j++)
            histogram[j] = histogram[j] / totalPixels;
    }
    
    return histogram;
}

int OtsuBinarizer::getThreshold(cv::Mat* inputImage, bool multiThreading, int threadCount)
{
    // Variable Declarations
    int threshold = 0;
    double allProbabilitySum = 0;
    size_t totalPixels = 0;

    // Code
    std::vector<double> histogram = getHistogram(inputImage, multiThreading, threadCount, &totalPixels);
    
    if (multiThreading)
    {   
        double* betweenClassVariances = new double[MAX_PIXEL_VALUE];

        #pragma omp parallel shared(allProbabilitySum, betweenClassVariances, totalPixels, histogram) num_threads(threadCount)
        {
            double firstClassProbability = 0, secondClassProbability = 0;
            double firstClassMean = 0, secondClassMean = 0, firstProbabilitySum = 0;

            int segmentSize = MAX_PIXEL_VALUE / threadCount;

            #pragma omp for schedule(static, segmentSize)
            for (int i = 0; i < MAX_PIXEL_VALUE; i++)
            {
                #pragma omp atomic
                allProbabilitySum += i * histogram[i];
                betweenClassVariances[i] = 0;
            }

            #pragma omp barrier

            #pragma omp for schedule(static, segmentSize)
            for (int j = 0; j < MAX_PIXEL_VALUE; j++)
            {
                firstClassProbability = 0;
			    firstProbabilitySum = 0;

                for (int k = 0; k <= j % MAX_PIXEL_VALUE; k++)
                {
                    firstClassProbability = firstClassProbability + histogram[k];
                    firstProbabilitySum += k * histogram[k];
                }

                secondClassProbability = 1 - firstClassProbability;

                firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
                secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

                betweenClassVariances[j] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
            }

            #pragma omp barrier

            #pragma omp single
            {
                double maxVariance = 0;

                for (int l = 0; l < MAX_PIXEL_VALUE; l++)
                {
                    if (betweenClassVariances[l] > maxVariance)
                    {
                        threshold = l;
                        maxVariance = betweenClassVariances[l];
                    }
                }
            }
        }

        delete[] betweenClassVariances;
        betweenClassVariances = nullptr;
    }
    else
    {
        //* Single Threaded
        double firstClassProbability = 0, secondClassProbability = 0;
        double firstClassMean = 0, secondClassMean = 0;
        double betweenClassVariance, maxVariance = 0;
        double firstProbabilitySum = 0;

        for (int i = 0; i < MAX_PIXEL_VALUE; i++)
            allProbabilitySum += i * histogram[i];

        for (int j = 0; j < MAX_PIXEL_VALUE; j++)
        {
            firstClassProbability = firstClassProbability + histogram[j];
            secondClassProbability = 1 - firstClassProbability;
            firstProbabilitySum = firstProbabilitySum + j * histogram[j];

            firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
            secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

            betweenClassVariance = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);

            if (betweenClassVariance > maxVariance)
            {
                threshold = j;
                maxVariance = betweenClassVariance;
            }
        }
    }

    // #if !RELEASE
    //     std::cout << std::endl << "Threshold : " << threshold << std::endl;
    // #endif
   
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

// Print Histogram
// double value = 0;
	// for (int i = 0; i < MAX_PIXEL_VALUE; i++) {
	// 	value = histogram[i];
	// 	printf("\tPixel value %d -> %.5f\n", i, value);
	// }
