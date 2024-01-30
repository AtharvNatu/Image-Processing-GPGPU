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

    #if !RELEASE
        cout << endl << "----------" << endl << "DEBUG MODE" << endl << "----------" << endl;
    #endif

    double cpuTime = cpuChangeDetector->detectChanges(
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/Dubai_1.jpg",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/Dubai_2.jpg",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/output"
    );

    cout << endl << "Time Required : " << cpuTime << " seconds" << endl;

    delete cpuChangeDetector;
    cpuChangeDetector = nullptr;

    return 0;
}
