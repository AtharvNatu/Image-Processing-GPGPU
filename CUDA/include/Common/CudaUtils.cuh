#include <cuda.h>
#include <iostream>

#include "Macros.hpp"

// Function Prototypes

//* CUDA Helper Functions

/// @brief cudaMalloc() Wrapper to allocate memory on device
/// @param devPtr Pointer to allocated device memory
/// @param size Requested allocation size in bytes
void cudaMemAlloc(void **devPtr, size_t size);

/// @brief cudaMemCpy() Wrapper which copies data between host and device
/// @param dst Destination memory address
/// @param src Source memory address
/// @param count Size in bytes to copy
/// @param kind Type of transfer
void cudaMemCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

/// @brief cudaFree() Wrapper which frees memory on the device
/// @param devPtr Device pointer to memory to free
void cudaMemFree(void **devPtr);

//* OpenCV Helper Functions

/// @brief Converts Image data to uchar3 pixel array
/// @param imageData Source Image Data
/// @param pixelArray Destination Pixel Array
/// @param size Size of source image
void convertImageToPixelArr(uchar_t *imageData, uchar3 *pixelArray, size_t size);

/// @brief Converts uchar3 pixel array to Image data
/// @param pixelArray Source Pixel Array
/// @param imageData Destination Image Data
/// @param size Size of source pixel array
void convertPixelArrToImage(uchar3 *pixelArray, uchar_t *imageData, size_t size);
