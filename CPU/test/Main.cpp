#include "../include/ChangeDetection/ChangeDetection.hpp"

int main(int argc, char **argv)
{
    // Code
    CPUChangeDetection *cpuChangeDetector = new CPUChangeDetection();

    #if !RELEASE
        cout << endl << "--------------------" << endl << "DEBUG MODE" << endl << "--------------------" << endl;
    #endif

    //* Single-Threaded
    // double cpuTime = cpuChangeDetector->detectChanges(
    //     "/home/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/petal-1.jpg",
    //     "/home/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/petal-2.jpg",
    //     "/home/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/output",
    //     false,
    //     0
    // );
    // cout << endl << "Time Required Without Multi-Threading : " << cpuTime << " seconds" << endl;

    //* Multi-threaded
    int threadCount = getThreadCount();

    double cpuTime = cpuChangeDetector->detectChanges(
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/Dubai_1.jpg",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/Dubai_2.jpg",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/output",
        true,
        threadCount
    );
    cout << endl << "Time Required Using Multi-Threading : Using " << threadCount << " Threads : " << cpuTime << " seconds" << endl;

    delete cpuChangeDetector;
    cpuChangeDetector = nullptr;

    return 0;
}
