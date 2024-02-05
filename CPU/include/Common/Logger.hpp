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
    private:
        FILE *logFile = nullptr;
        string getCurrentTime(void);

    protected:
        Logger(void);
        static Logger* _logger;

    public:
        //* Non-cloneable
        Logger(Logger &obj) = delete;

        //* Non-assignable
        void operator = (const Logger &) = delete;

        Logger(const string file);
        ~Logger(void);

        // Member Function Declarations
        static Logger* getInstance(const string file);
        void printLog(const char* fmt, ...);
        void deleteInstance(void);
};

