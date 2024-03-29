#ifndef _CLFW_HEADER_HPP_
    
    #define _CLFW_HEADER_HPP_

    // Standard Headers
    #include <iostream>
    #include <cstdlib>
    #include <cstdarg>
    #include <filesystem>
    #include <cstring>
    #include <cmath>
    #include <opencv2/opencv.hpp>
    
    // CLFW Headers
    #include "CLFWTypes.hpp"

    class CLFW
    {
        // Member Variable Declarations
        private:

            // CLFW Kernel Data Types
            CLFWTypes *clfwTypes = nullptr;

            //* Device Enumeration
            cl_platform_id *oclPlatforms = NULL;
            cl_device_id *oclDevices = NULL;
            cl_uint oclNumPlatforms = 0, oclNumDevices = 0;
            char oclPlatformInfo[512], oclDevProp[1024];

            //* Device Selection
            cl_platform_id oclPlatformId;
            cl_device_id oclDeviceId;
            int user_platform, user_device;

            //* Device Properties
            cl_ulong oclMemSize;
            cl_uint oclComputeUnits;

            cl_context oclContext;
            cl_command_queue oclCommandQueue;
            const char* oclKernelSourceCode = NULL;
            cl_program oclProgram;
            cl_kernel oclKernel;
            cl_int oclResult;
            cl_mem oclBuffer;
            cl_image_format oclImageFormat;
            cl_image_desc oclImageDescriptor;
            cl_event timeEvent;
            cl_ulong startTime, endTime;

        public:

            CLFW(void);
            ~CLFW();

            // Member Function Prototypes
            void initialize(void);
            void uninitialize(void);

            void oclExecStatus(cl_int oclResult);
            std::string oclGetErrorString(cl_int oclResult);

            void oclGetPlatforms(void);
            void oclSetPlatform(int oclPlatform);

            void oclGetDevices(void);
            void oclSetDevice(int oclDevice);

            void oclGetDeviceProperties(void);
            size_t oclGetDeviceMaxWorkGroupSize(void);

            void oclCreateImage(cl_mem *devPtr, cl_mem_flags flags, size_t imageWidth, size_t imageHeight, unsigned char *imagePixels);
            void oclReadImage(cl_mem *devImagePtr, int imageWidth, int imageHeight, unsigned char *imagePixels);

            void oclCreateContext(void);
            void oclCreateCommandQueue(void);
            void oclCreateProgram(const char* oclKernelSource);
            void oclCreateKernel(const char* oclKernelName, const char *oclKernelArgTypes,...);
            size_t getGlobalWorkSize(int localSize, unsigned int globalSize);
            double oclExecuteKernel(size_t oclGlobalWorkSize, size_t oclLocalWorkSize, int workDimensions);
            double oclExecuteKernel(size_t *oclGlobalWorkSize, size_t *oclLocalWorkSize, int workDimensions);

            void hostMemAlloc(void** hostPtr, std::string hostType, size_t hostSize);

            template <typename T>
            void hostMemFree(T **hostPtr);

            cl_mem oclCreateBuffer(int flag, size_t oclDataSize);
            void oclWriteBuffer(cl_mem oclDataBuffer, size_t oclDataSize, void* hostPtr);
            void oclFillBuffer(cl_mem oclDataBuffer, const void *pattern, size_t patternSize, size_t offset, size_t size);
            void oclReadBuffer(cl_mem oclDataBuffer, size_t oclDataSize, void* hostPtr);
            void oclReleaseBuffer(cl_mem oclDataBuffer);
    };

#endif // _CLFW_HEADER_HPP_


