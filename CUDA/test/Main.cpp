#include "../include/ChangeDetection/CudaChangeDetection.cuh"

int main(int argc, char **argv)
{
    // Code
    CudaChangeDetection *cudaChangeDetector = new CudaChangeDetection();

    std::cout << std::endl << "--------------------" << std::endl << "DEBUG MODE" << std::endl << "--------------------" << std::endl;

    if (std::string(argv[4]) == "ST")
    {
        //* Single-Threaded
        double cpuTime = cudaChangeDetector->detectChanges(
            std::string(argv[1]),
            std::string(argv[2]),
            std::string(argv[3]),
            true,
            false,
            0
        );
        
        std::cout << std::endl << "Time Required Using Single Thread : " << cpuTime << " seconds" << std::endl;
    }
    else if (std::string(argv[4]) == "MT")
    {
        //* Multi-threaded
        int threadCount = getThreadCount();
        std::cout << std::endl << "Thread Count : " << threadCount << std::endl;

        double cpuTime = cpuChangeDetector->detectChanges(
            std::string(argv[1]),
            std::string(argv[2]),
            std::string(argv[3]),
            false,
            true,
            threadCount
        );
        
        std::cout << std::endl << "Time Required Using Multi-Threading : Using " << threadCount << " Threads : " << cpuTime << " seconds" << std::endl;
    }
    
    delete cpuChangeDetector;
    cpuChangeDetector = nullptr;

    return 0;
}
