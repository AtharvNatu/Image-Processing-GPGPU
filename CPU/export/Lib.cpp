#include "../include/ChangeDetection/ChangeDetection.hpp"

extern "C" double cpuChangeDetection(void)
{
    // Code
    CPUChangeDetection *cpuChangeDetector = new CPUChangeDetection();

    double cpuTime = cpuChangeDetector->detectChanges(
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/1024_old.png",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/1024_new.png",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/output"
    );

    delete cpuChangeDetector;
    cpuChangeDetector = nullptr;

    return cpuTime;
}
