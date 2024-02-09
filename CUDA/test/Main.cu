#include "../include/ChangeDetection/CudaChangeDetection.cuh"


int main(int argc, char **argv)
{
    // Code
    CudaChangeDetection *cudaChangeDetector = new CudaChangeDetection();

    std::cout << std::endl << "--------------------" << std::endl << "DEBUG MODE" << std::endl << "--------------------" << std::endl;

    double gpuTime = cudaChangeDetector->detectChanges(
        std::string(argv[1]),
        std::string(argv[2]),
        std::string(argv[3]),
        false
    );
        
    std::cout << std::endl << "Time Required Using CUDA : "<< gpuTime << " seconds" << std::endl;
    
    delete cudaChangeDetector;
    cudaChangeDetector = nullptr;

    return 0;
}
