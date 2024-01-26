#include "../include/Common/Logger.hpp"
#include "../include/ChangeDetection/ChangeDetection.hpp"

int main(int argc, char **argv)
{
    initializeLog();
    printLog("Initial Log Test ...");

    cpuDetectChanges(string(argv[1]), "");

    return 0;
}
