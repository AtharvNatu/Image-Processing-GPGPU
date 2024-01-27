#include "../../include/Common/Timer.hpp"

// Examples
// Windows
// double KernelProcessStartTime = 0, KernelProcessEndTime = 0, UserProcessTime = 0;

// double start = GetCPUTime(&KernelProcessStartTime);
// {
//     double sum = 0, add = 1;
//     int iterations = 1000 * 1000 * 1000;
//     for (int i = 0; i < iterations; i++)
//     {
//         sum += add;
//         add /= 2.0;
//     }
// }
// double end = GetCPUTime(&KernelProcessEndTime);

// cout << endl << "Execution Time : User : " << end - start << endl;
// cout << endl << "Execution Time : Kernel : " << KernelProcessEndTime - KernelProcessStartTime << endl;

// double elapsed = (KernelProcessEndTime - KernelProcessStartTime) + (end - start);
// cout << endl << "Time in seconds : " << elapsed << endl;

// *Nix
//  clock_t start = clock();
//     {
//         double sum = 0, add = 1;
//         int iterations = 1000 * 1000 * 1000;
//         for (int i = 0; i < iterations; i++)
//         {
//             sum += add;
//             add /= 2.0;
//         }
//     }
//     clock_t end = clock();
//     double elapsed1 = 1000.0 * (end - start) / CLOCKS_PER_SEC;
//     cout << endl << "Time in milliseconds (Method-1) : " << elapsed1 << endl;
//     cout << endl << "Time in seconds (Method-1) : " << elapsed1 / 1000.0 << endl;

// Function Definitions
#if (OS == 1)

    double getTime(void)
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
            double time = 0;

            // User System Time
            if (FileTimeToSystemTime(&UserTime, &UserSystemTime) != -1)
            {
                time =  (double)UserSystemTime.wHour * 3600.0 +
                        (double)UserSystemTime.wMinute * 60.0 +
                        (double)UserSystemTime.wSecond +
                        (double)UserSystemTime.wMilliseconds / 1000;
            }

            // Kernel System Time
            if (FileTimeToSystemTime(&KernelTime, &KernelSystemTime) != -1)
            {
                *KernelProcessTime = (double)KernelSystemTime.wHour * 3600.0 +
                                    (double)KernelSystemTime.wMinute * 60.0 +
                                    (double)KernelSystemTime.wSecond +
                                    (double)KernelSystemTime.wMilliseconds / 1000;
            }

            time = (double)(UserTime.dwLowDateTime |
                ((unsigned long long)UserTime.dwHighDateTime << 32)) * 0.0000001;

            return time;
        }
        else
        {
            // ErrorExit(TEXT("GetProcessTimes"));
            return -1;
        }
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


