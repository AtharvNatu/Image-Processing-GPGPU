#include "Macros.hpp"

#if (OS == 1)

    #include <windows.h>
    #include <strsafe.h>
    #include <processthreadsapi.h>

    #pragma comment(lib, "user32.lib")
    #pragma comment(lib, "kernel32.lib")

    // Function Declarations
    double getClockTime(void);
    double getExecutionTime(double start, double end);

#elif (OS == 2 || OS == 3)
    
    #include <ctime>

    // Function Declarations
    clock_t getClockTime(void);
    double getExecutionTime(clock_t start, clock_t end);

#endif




