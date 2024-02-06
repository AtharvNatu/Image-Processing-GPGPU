#include "../include/ChangeDetection/ChangeDetection.hpp"

#if (OS == 1)
    #include <windows.h>

    

#elif (OS == 2)

    extern "C" double cpuChangeDetection(std::string oldImagePath, std::string newImagePath, std::string outputImagePath, bool grayscale, bool multiThreading)
    {
        // Code
        CPUChangeDetection *cpuChangeDetector = new CPUChangeDetection("./logs/IPUG.log");
        double cpuTime = 0;

        #if !RELEASE
            std::cout << std::endl << "--------------------" << std::endl << "DEBUG MODE" << std::endl << "--------------------" << std::endl;
        #endif

        //* Multi-threaded
        if (multiThreading)
        {
            int threadCount = getThreadCount();

            cpuTime = cpuChangeDetector->detectChanges(
                oldImagePath,
                newImagePath,
                outputImagePath,
                grayscale,
                true,
                threadCount  
            );

            std::cout << std::endl << "Time Required Using Multi-Threading : Using " << threadCount << " Threads : " << cpuTime << " seconds" << std::endl;
        }

        //* Single-Threaded
        // else
        // {
        //     cpuTime = cpuChangeDetector->detectChanges(
        //         oldImagePath,
        //         newImagePath,
        //         outputImagePath,
        //         grayscale,
        //         false,
        //         0  
        //     );

        //     std::cout << std::endl << "Time Required Using Single Thread : " << cpuTime << " seconds" << std::endl;
        // }
        
        delete cpuChangeDetector;
        cpuChangeDetector = nullptr;

        return cpuTime;
    }

#endif

