#include "tk_time.h"
#include <stdatomic.h>
#include <stdint.h>

#ifdef _WIN32
    #include "windows.h"
#else
    #include <time.h>
#endif

uint64_t tk_get_now_ns(void) {
#ifdef _WIN32

    static LARGE_INTEGER frequency;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    return (uint64_t)((counter.QuadPart * 1000000000LL) / frequency.QuadPart);
#else

    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}
