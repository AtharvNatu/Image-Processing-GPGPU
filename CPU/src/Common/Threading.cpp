#include "../../include/Common/Threading.hpp"

int getThreadCount(void)
{
    // Code
    int count = 1;

    #pragma omp parallel
    {
        #pragma omp single
        count = omp_get_num_threads();
    }

    return count;
}

