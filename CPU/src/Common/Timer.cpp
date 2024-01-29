#include "../../include/Common/Timer.hpp"

// Function Definitions
#if (OS == 1)

    double getClockTime(void)
    {
        // Variables
        FILETIME CreationTime, ExitTime, KernelTime, UserTime;
        
        // Code
        if (GetProcessTimes(
            GetCurrentProcess(), 
            &CreationTime,
            &ExitTime,
            &KernelTime,
            &UserTime) != -1)
        {
            SYSTEMTIME UserSystemTime, KernelSystemTime;
            double userTime = 0, systemTime = 0;

            // User System Time
            if (FileTimeToSystemTime(&UserTime, &UserSystemTime) != -1)
            {
                userTime =  (double)UserSystemTime.wHour * 3600.0 +
                            (double)UserSystemTime.wMinute * 60.0 +
                            (double)UserSystemTime.wSecond +
                            (double)UserSystemTime.wMilliseconds / 1000;
            }

            // Kernel System Time
            if (FileTimeToSystemTime(&KernelTime, &KernelSystemTime) != -1)
            {
                systemTime = (double)KernelSystemTime.wHour * 3600.0 +
                            (double)KernelSystemTime.wMinute * 60.0 +
                            (double)KernelSystemTime.wSecond +
                            (double)KernelSystemTime.wMilliseconds / 1000;
            }

            // time = (double)(UserTime.dwLowDateTime |
            //     ((unsigned long long)UserTime.dwHighDateTime << 32)) * 0.0000001;

            return userTime + systemTime;
        }
        else
        {
            // ErrorExit(TEXT("GetProcessTimes"));
            return -1;
        }
    }

    double getExecutionTime(double start, double end)
    {
        return double(end - start);
    }

#elif (OS == 2 || OS == 3)

    clock_t getClockTime(void)
    {
        return clock();
    }

    double getExecutionTime(clock_t start, clock_t end)
    {
        return double(end - start) / CLOCKS_PER_SEC;
    }

#endif


