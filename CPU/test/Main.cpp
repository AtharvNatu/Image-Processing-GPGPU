#include "../include/Common/Logger.hpp"
#include "../include/ChangeDetection/ChangeDetection.hpp"

int main(int argc, char **argv)
{
    // Code
    Logger *logger = nullptr;

    logger = new Logger();
    
    logger->initialize();
    logger->printLog("C++ Log Test : Linux...");

    double cpuTime = cpuDetectChanges(
        "/home/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/petal-1.jpg",
        "/home/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/petal-2.jpg",
        "/home/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/output"
    );

    cout << endl << "Time Required : " << cpuTime << " seconds" << endl;

    // int threads = get_num_threads();
    // cout << endl << "Threads = " << threads << endl;
    // #pragma omp parallel for num_threads(threads)
    // for (int i = 1; i <= 100; i++) 
    // {
    //     int tid = omp_get_thread_num();
    //     printf("The thread %d  executes i = %d\n", tid, i);
    // }

    logger->uninitialize();
    logger = nullptr;

    return 0;
}
