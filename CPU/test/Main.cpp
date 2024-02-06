#include "../include/ChangeDetection/ChangeDetection.hpp"

int main(int argc, char **argv)
{
    // Code
    CPUChangeDetection *cpuChangeDetector = new CPUChangeDetection("./logs/IPUG.log");

    #if !RELEASE
        std::cout << std::endl << "--------------------" << std::endl << "DEBUG MODE" << std::endl << "--------------------" << std::endl;
    #endif

    //* Single-Threaded
    // double cpuTime = cpuChangeDetector->detectChanges(
    //     std::string(argv[1]),
    //     std::string(argv[2]),
    //     std::string(argv[3]),
    //     true,
    //     false,
    //     0
    // );
    
    // std::cout << std::endl << "Time Required Using Single Thread : " << std::cpuTime << " seconds" << std::endl;

    //* Multi-threaded
    int threadCount = getThreadCount();

    double cpuTime = cpuChangeDetector->detectChanges(
        std::string(argv[1]),
        std::string(argv[2]),
        std::string(argv[3]),
        true,
        true,
        threadCount
    );
    
    std::cout << std::endl << "Time Required Using Multi-Threading : Using " << threadCount << " Threads : " << cpuTime << " seconds" << std::endl;

    delete cpuChangeDetector;
    cpuChangeDetector = nullptr;

    return 0;
}
