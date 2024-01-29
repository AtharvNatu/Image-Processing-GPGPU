#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #define OS 1
#elif defined(__linux)
    #define OS 2
#elif defined(__APPLE__)
    #define OS 3
#endif

#if (OS == 1)
    #include <windows.h>
#endif

enum ERRORS
{
    LOG_ERROR = -1,
    OPENCV_ERROR = -2,
    FILE_ERROR = -3
};

// Typedefs
typedef unsigned char uchar_t;
