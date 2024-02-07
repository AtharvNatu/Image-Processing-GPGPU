#include "../include/ChangeDetection/ChangeDetection.hpp"

#if (OS == 1)

    BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) 
    {
        switch (ul_reason_for_call) 
        {
            case DLL_PROCESS_ATTACH:
            case DLL_THREAD_ATTACH:
            case DLL_THREAD_DETACH:
            case DLL_PROCESS_DETACH:
                break;
        }
        return TRUE;
    }

    extern "C" __declspec(dllexport) double cpuChangeDetection(std::string oldImagePath, std::string newImagePath, std::string outputImagePath, bool grayscale, bool multiThreading)
    {
        // Variable Declarations
        double cpuTime = 0;

        // Code
        CPUChangeDetection *cpuChangeDetector = new CPUChangeDetection("IPUG.log");

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
        else
        {
            cpuTime = cpuChangeDetector->detectChanges(
                oldImagePath,
                newImagePath,
                outputImagePath,
                grayscale,
                false,
                0  
            );

            std::cout << std::endl << "Time Required Using Single Thread : " << cpuTime << " seconds" << std::endl;
        }
        
        delete cpuChangeDetector;
        cpuChangeDetector = nullptr;

        return cpuTime;
    }

#else

    extern "C" double cpuChangeDetection(std::string oldImagePath, std::string newImagePath, std::string outputImagePath, bool grayscale, bool multiThreading)
    {
        // Variable Declarations
        double cpuTime = 0;

        // Code
        CPUChangeDetection *cpuChangeDetector = new CPUChangeDetection("IPUG.log");

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
        else
        {
            cpuTime = cpuChangeDetector->detectChanges(
                oldImagePath,
                newImagePath,
                outputImagePath,
                grayscale,
                false,
                0  
            );

            std::cout << std::endl << "Time Required Using Single Thread : " << cpuTime << " seconds" << std::endl;
        }
        
        delete cpuChangeDetector;
        cpuChangeDetector = nullptr;

        return cpuTime;
    }

#endif

