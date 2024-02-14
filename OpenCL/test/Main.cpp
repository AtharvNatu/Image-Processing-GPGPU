#include "../include/ChangeDetection/OpenCLChangeDetection.hpp"

int main(int argc, char **argv)
{
    // Code
    OpenCLChangeDetection *oclChangeDetector = new OpenCLChangeDetection();

    std::cout << std::endl << "--------------------" << std::endl << "DEBUG MODE" << std::endl << "--------------------" << std::endl;

    // double gpuTime = oclChangeDetector->detectChanges(
    //     std::string(argv[1]),
    //     std::string(argv[2]),
    //     std::string(argv[3]),
    //     false
    // );
    double gpuTime = oclChangeDetector->detectChanges(
        "F:\\Internship\\Images\\10000_old.png",
        "F:\\Internship\\Images\\10000_new.png",
        "F:\\Internship\\Images",
        false
    );

    std::cout << std::endl << "Time Required Using CUDA : "<< gpuTime << " seconds" << std::endl;
    
    delete oclChangeDetector;
    oclChangeDetector = nullptr;

    return 0;
}
