#include <cuda.h>
#include <iostream>

#include "Macros.hpp"

// Function Prototypes

// CUDA Helper Functions
void cudaMemAlloc(void **devPtr, size_t size);
void cudaMemCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
void cudaMemFree(void **devPtr);

// OpenCV Image Data to uchar
void convertImageToPixelArr(uchar3 *pixelArray, uchar_t *imageData, size_t size);
void convertPixelArrToImage(uchar3 *pixelArray, uchar_t *imageData, size_t size);
