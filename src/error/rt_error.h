#ifndef RTERR_H
#define RTERR_H

#include <stdio.h>
#include <errno.h>

typedef enum rt_errc {
    RT_OK = 0,
    RT_EINVAL = 1,
    RT_EOOM = 2,
    RT_EIO = 3,
    RT_ESTATE = 4,
    RT_EINTERNAL = 5
} rt_errc;

struct rt_err_status {
    enum rt_errc code;
    int sys_errno;
    int line;
    const char *file;
    const char *func;
    char msg[256];
};

int rt_err_set(rt_errc code, int sys_errno,
               const char *file, int line, const char *func,
               const char *fmt, ...);

void rt_err_clear(void);
void rt_err_print(FILE *out);
const struct rt_err_status *rt_err_last(void);

// Only set error, does not return (as oppose to RT_FAIL)
#define RT_SET(code, fmt, ...) \
    ((void)rt_err_set((code), 0, \
        __FILE__, __LINE__, __func__, \
        (fmt), ##__VA_ARGS__))

#define RT_SET_ERRNO(code, fmt, ...) \
    do { \
        int _e = errno; \
        (void)rt_err_set((code), _e, \
            __FILE__, __LINE__, __func__, \
            (fmt), ##__VA_ARGS__); \
    } while (0)

// Set error and return immediately
#define RT_FAIL(code, fmt, ...) \
    return rt_err_set((code), 0, \
        __FILE__, __LINE__, __func__, \
        (fmt), ##__VA_ARGS__)

// preserve errno
#define RT_FAIL_ERRNO(code, fmt, ...) \
    do { \
        int _e = errno; \
        return rt_err_set((code), _e, \
            __FILE__, __LINE__, __func__, \
            (fmt), ##__VA_ARGS__); \
    } while (0)

#define RT_CHECK(expr) \
    do { \
        int _rc = (expr); \
        if (_rc < 0) \
            return _rc; \
    } while (0)

#define RT_CHECK_GOTO(expr, rc, label) \
    do {                               \
        (rc) = (expr);                 \
        if ((rc) < 0)                  \
            goto label;                \
    } while (0)

#endif
