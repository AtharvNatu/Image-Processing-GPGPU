#include "../include/ChangeDetection/ChangeDetection.hpp"

// int getThreadCount(void)
// {
//     // Code
//     int count = 1;

//     #pragma omp parallel
//     {
//         #pragma omp single
//         count = omp_get_num_threads();
//     }

//     return count;
// }

int main(int argc, char **argv)
{
    // Code
    CPUChangeDetection *cpuChangeDetector = new CPUChangeDetection();

    #if RELEASE
        logger->printLog("Application Running in RELEASE Mode...");
    #else
        cout << endl << "----------" << endl << "DEBUG MODE" << endl << "----------" << endl;
    #endif

    double cpuTime = cpuChangeDetector->detectChanges(
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/1024_old.png",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/1024_new.png",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/output"
    );

    cout << endl << "Time Required : " << cpuTime << " seconds" << endl;

    delete cpuChangeDetector;
    cpuChangeDetector = nullptr;

    return 0;
}
