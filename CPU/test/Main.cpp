#include "../include/Common/Logger.hpp"
#include "../include/ChangeDetection/ChangeDetection.hpp"

int main(int argc, char **argv)
{
    // Code
    Logger *logger = nullptr;

    logger = new Logger();
    
    logger->initialize();
    logger->printLog("C++ Log Test : macOS...");

    double cpuTime = cpuDetectChanges(
        "F:\\Internship\\Image-Processing-GPGPU\\CPU\\images\\input\\1024_old.png",
        "F:\\Internship\\Image-Processing-GPGPU\\CPU\\images\\input\\1024_new.png",
        "F:\\Internship\\Image-Processing-GPGPU\\CPU\\images\\output"
    );

    cout << endl << "Time Required : " << cpuTime << " seconds" << endl;

    logger->uninitialize();
    logger = nullptr;

    return 0;
}
