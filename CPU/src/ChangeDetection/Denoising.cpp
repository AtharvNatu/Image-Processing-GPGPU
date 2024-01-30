#include "../../include/ChangeDetection/Denoising.hpp"

Denoising::Denoising(void)
{
    // Code
    kernel = nullptr;
}

void Denoising::gaussianBlur(cv::Mat *inputImage, cv::Mat *outputImage)
{
    // Variable Declarations
    float kernelSum = 0.0f, sigma = 1.0f;

    // Code
    int imageWidth = inputImage->cols;
    int imageHeight = inputImage->rows;
    int imageSize = imageHeight * imageWidth * sizeof(uchar_t);

    // Create Gaussian Kernel(Mask)
    kernel = new float[GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE];
    int kernelRadius = GAUSSIAN_KERNEL_SIZE / 2;

    for (int i = -kernelRadius; i <= kernelRadius; i++)
    {
        for (int j = -kernelRadius; j <= kernelRadius; j++)
        {
            int index = (i + kernelRadius) * kernelRadius + (j + kernelRadius);
            kernel[index] = exp(-(i * i + j + j) / (2.0f * sigma * sigma));
            kernelSum = kernelSum + kernel[index];
        }
    }

    for (int i = 0; i < GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE; i++)
        kernel[i] = kernel[i] / kernelSum;

    __gaussianBlurKernel(inputImage->data, outputImage->data, imageWidth, imageHeight, kernel);
}

void Denoising::__gaussianBlurKernel(uchar_t *inputImage, uchar_t *outputImage, int imageWidth, int imageHeight, float *kernel)
{
    // Variable Declarations
    float blurPixel = 0.0f;
    int kernelRadius = GAUSSIAN_KERNEL_SIZE / 2;

    // Code
    for (int i = -kernelRadius; i <= kernelRadius; i++)
    {
        for (int j = -kernelRadius; j <= kernelRadius; j++)
        {
            int x_offset = x + i;
            int y_offset = y + j;

            if (x_offset >= 0 && x_offset < imageWidth && y_offset >= 0 && y_offset < imageHeight)
            {
                int input_index = y_offset * imageWidth + x_offset;
                int kernel_index = (i + kernelRadius) * GAUSSIAN_KERNEL_SIZE + (j + kernelRadius);
                blurPixel = blurPixel + static_cast<float>(inputImage[input_index]) * kernel[kernel_index];
            }
        }
    }

    outputImage[y * imageWidth + x] = static_cast<unsigned char>(blurPixel);

}

Denoising::~Denoising(void)
{
    // Code
    if (kernel)
    {
        delete[] kernel;
        kernel = nullptr;
    }
}
