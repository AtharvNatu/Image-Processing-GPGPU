#pragma once

#include <cuda.h>
#include <iostream>

#include "Macros.hpp"

class CudaUtils
{
    public:

        //* CUDA Helper Functions For Memory Management With Error Checking
        //*----------------------------------------------------------------------------------------*
        
        /// @brief cudaMalloc() Wrapper to allocate memory on device
        /// @param devPtr Pointer to allocated device memory
        /// @param size Requested allocation size in bytes
        void memAlloc(void **devPtr, size_t size);

        /// @brief cuMemSet() Wrapper to initialize memory on device
        /// @param devPtr Pointer to allocated device memory
        /// @param value Value to set for each byte of specified memory
        /// @param count Size in bytes to set
        void memSet(void *devPtr, int value, size_t count);

        /// @brief cudaMemCpy() Wrapper which copies data between host and device
        /// @param dst Destination memory address
        /// @param src Source memory address
        /// @param count Size in bytes to copy
        /// @param kind Type of transfer
        void memCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

        /// @brief cudaFree() Wrapper which frees memory on the device
        /// @param devPtr Device pointer to memory to free
        void memFree(void **devPtr);

        //*----------------------------------------------------------------------------------------*

        //* CUDA Events Related Wrappers With Error Checking
        //*----------------------------------------------------------------------------------------*
        
        /// @brief cudaEventCreate() Wrapper
        /// @param event Newly created event
        void createEvent(cudaEvent_t *event);

        /// @brief cudaEventCreate() Wrapper
        /// @param event Event to record
        /// @param stream Stream in which to record event
        void recordEvent(cudaEvent_t event, cudaStream_t stream);

        /// @brief cudaEventSynchronize() Wrapper
        /// @param event Event to wait for
        void syncEvent(cudaEvent_t event);

        /// @brief cudaEventCreate() Wrapper
        /// @param ms Time between start and end in ms
        /// @param event Starting event
        /// @param event Ending event
        void getEventElapsedTime(double *ms, cudaEvent_t start, cudaEvent_t end);

        /// @brief cudaEventDestroy() Wrapper
        /// @param event Event to destroy
        void destroyEvent(cudaEvent_t event);

        //*----------------------------------------------------------------------------------------*

        //* OpenCV Helper Functions
        //*----------------------------------------------------------------------------------------*
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
        //*----------------------------------------------------------------------------------------*

};

