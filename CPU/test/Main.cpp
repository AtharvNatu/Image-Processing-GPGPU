#include "../include/Common/Logger.hpp"
#include "../include/ChangeDetection/ChangeDetection.hpp"

int main(int argc, char **argv)
{
    // Code
    Logger *logger = nullptr;

    logger = new Logger();
    
    logger->initialize();
    logger->printLog("C++ Log Test : macOS...");

    cpuDetectChanges(
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/1024_old.png",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/input/1024_new.png",
        "/Users/atharv/Desktop/Internship/Code/Image-Processing-GPGPU/CPU/images/output"
    );

    logger->uninitialize();
    logger = nullptr;

    return 0;
}
