#include "../../include/ChangeDetection/Denoising.hpp"

// Method Definitions
void Denoising::getWindow(uchar_t* imageData, uchar_t* window, int row, int column, int width, int size)
{
    // Code
    int mid = (int) (size - 1) / 2;

    for (int i = -mid; i < mid + 1; i++)
    {
        for (int j = -mid; j < mid + 1; j++)
        {
            *window = *(imageData + row * width + i + column + j);
            window++;
        }
    }
}

void Denoising::subtractKernels(uchar_t* k1, uchar_t* k2, double* result, int size)
{
    // Code
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            result[i * size + j] = ((double) k1[i * size + j]) - ((double) k2[i * size + j]);
    }
}

double Denoising::computeKernelNorm(double* kernel, int size)
{
    // Code
    double sum = 0.0;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            sum = sum + pow(kernel[i * size + j], 2.0);
    }

    return sqrt(sum);
}

void Denoising::getGaussianKernel(double* kernel, int size)
{
    // Variable Declarations
    double mid = (size - 1) / 2.0;
    double kernelSum = 0;

    // Code
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            double x = i - mid;
            double y = j - mid;

            //* Gaussian Distribution Evaluation
            kernel[i * size + j] = exp(-(x * x + y + y) / (2 * STDDEV * STDDEV));
            kernelSum = kernelSum + kernel[i * size + j];
        }
    }

    //* Normalize
    for (int i = 0; i < size * size; i++)
        kernel[i] = kernel[i] / kernelSum;

}

void Denoising::__gaussianFilter(uchar_t *image, int imageWidth, int imageHeight)
{
    // Variable Declarations
    int mid = (int) (GAUSSIAN_KERNEL_SIZE - 1) / 2;
    uchar_t blurPixel = 0;

    uchar_t* window = new uchar_t[GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE];
    double* gaussianKernel = new double[GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE];

    // Code
    getGaussianKernel(gaussianKernel, mid);

    for (int i = mid; i < imageHeight - mid; i++)
    {
        for (int j = mid; j < imageWidth - mid; j++)
        {
            double sum = 0.0;

            getWindow(image, window, i, j, imageWidth, GAUSSIAN_KERNEL_SIZE);

            for (int k = 0; k < GAUSSIAN_KERNEL_SIZE; k++)
            {
                for (int l = 0; l < GAUSSIAN_KERNEL_SIZE; l++)
                    sum += gaussianKernel[k * GAUSSIAN_KERNEL_SIZE + l] * ((double) window[k * GAUSSIAN_KERNEL_SIZE + l]);
            }

            // Bound between 0-255
            if (sum > 255)
                blurPixel = 255;
            else if (sum > 0)
                blurPixel = (uchar_t) round(sum);

            image[i * imageWidth + j] = blurPixel;
        }
    }

    delete[] gaussianKernel;
    gaussianKernel = nullptr;

    delete[] window;
    window = nullptr;
    
}

void Denoising::__nonLocalMeansFilter(uchar_t *image, int imageWidth, int imageHeight)
{
    // Variable Delcarations
    int midWindow = (int) (WINDOW_SIZE - 1) / 2;
    int midSimWindow = (int) (SIMILARITY_WINDOW_SIZE - 1) / 2;
    

    uchar_t* window =  (uchar_t*) malloc(WINDOW_SIZE * WINDOW_SIZE * sizeof(uchar_t));
    uchar_t* simWindow =  (uchar_t*) malloc(WINDOW_SIZE * WINDOW_SIZE * sizeof(uchar_t));
    double* resultWindow =  (double*) malloc(WINDOW_SIZE * WINDOW_SIZE * sizeof(double));
    
    // Code
    for (int i = midWindow; i < imageHeight - midWindow; i++)
    {
        for (int j = midWindow; j < imageWidth - midWindow; j++)
        {
            double sum = 0.0, normalizationFactor = 0.0;

            getWindow(image, window, i, j, imageHeight, WINDOW_SIZE);

            int uMin = MAX(i - midSimWindow, midWindow);
            int uMax = MIN(i + midSimWindow, imageHeight - midWindow);
            int vMin = MAX(j - midSimWindow, midWindow);
            int vMax = MIN(j + midSimWindow, imageWidth - midWindow);

            for (int u = uMin; u < uMax + 1; u++)
            {
                for (int v = vMin; v < vMax + 1; v++)
                {
                    getWindow(image, simWindow, u, v, imageHeight, WINDOW_SIZE);

                    subtractKernels(window, simWindow, resultWindow, WINDOW_SIZE);

                    double normValue = computeKernelNorm(resultWindow, WINDOW_SIZE);
                    double similarity = exp(-normValue / pow(SIGMA, 2.0));

                    normalizationFactor += similarity;
                    sum += similarity * image[u * imageWidth + v];
                }
            }

            double result = sum / normalizationFactor;
            uchar_t blurPixel = 0;

            if (result > 255)
                blurPixel = 255;
            else if (result > 0)
                blurPixel = (uchar_t) round(result);

            image[i * imageWidth + j] = blurPixel;
        }
    }

    free(resultWindow);
    resultWindow = NULL;

    free(simWindow);
    simWindow = NULL;

    free(window);
    window = NULL;
}

void Denoising::gaussianDenoising(cv::Mat *image)
{
    // Code
    int imageWidth = image->cols;
    int imageHeight = image->rows;

    __gaussianFilter(image->data, imageWidth, imageHeight);
}

void Denoising::nlmDenoising(cv::Mat *image)
{
    // Code
    int imageWidth = image->cols;
    int imageHeight = image->rows;

    __nonLocalMeansFilter(image->data, imageWidth, imageHeight);
}

