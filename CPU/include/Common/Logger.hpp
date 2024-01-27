#pragma once

#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdarg>

#include "Macros.hpp"

#ifndef _STD_NS_    // namespace std
    #define _STD_NS_
    using namespace std;
#endif              // namespace std


class Logger
{
    // Member Variables
    private:
        FILE *logFile = nullptr;

    // Member Function Declarations
    public:
        void initialize(void);
        void printLog(const char* fmt, ...);
        string getCurrentTime(void);
        void uninitialize(void);
};



