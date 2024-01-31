#include "../../include/Common/Logger.hpp"

Logger::Logger(void)
{
    // Code
    logFile = fopen("./logs/Log.txt", "a+");
    if (logFile == nullptr)
    {
        cerr << endl << "Failed To Open Log File : logs/Log.txt ... Exiting !!!";
        exit(LOG_ERROR);
    }
}

void Logger::printLog(const char* fmt, ...)
{
    // Variable Declarations
    va_list argList;

    // Code
    if (logFile == nullptr)
        return;

    // Print Current Time To File
    fprintf(logFile, "%s", getCurrentTime().c_str());
    fprintf(logFile, "\t");

    // Print Log Data
    va_start(argList, fmt);
    {
        vfprintf(logFile, fmt, argList);
    }
    va_end(argList);

    fprintf(logFile, "\n");
}

string Logger::getCurrentTime(void)
{
    // Code
    time_t currentTime = chrono::system_clock::to_time_t(chrono::system_clock::now());
    string strTime(30, '\0');
    strftime(&strTime[0], strTime.size(), "%d/%m/%Y | %H:%M:%S", localtime(&currentTime));
    return strTime;
}

Logger::~Logger(void)
{
    // Code
    if (logFile)
    {
        printLog("Log File Closed ...");
        fclose(logFile);
        logFile = nullptr;
    }
}
