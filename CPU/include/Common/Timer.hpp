#include "Macros.hpp"

// Common Function Pointer Declration
void (*pFn)(int);

#if (OS == 1)

    #include <windows.h>
    #include <strsafe.h>
    #include <processthreadsapi.h>

    #pragma comment(lib, "user32.lib")
    #pragma comment(lib, "kernel32.lib")

    // Function Declarations
    double getExecutionTime(void);

#elif (OS == 2 || OS == 3)
    
    #include <time.h>

    // Function Declarations
    clock_t getClockTime(void);
    double getExecutionTime(clock_t start, clock_t end);

#endif




