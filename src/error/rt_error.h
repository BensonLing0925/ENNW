#ifndef RTERR_H
#define RTERR_H

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
    const char* file;
    const char* func;
    char msg[256]; 
};

int rt_err_set(rt_errc code, int sys_errno,
               const char* file, int line, const char* func,
               const char* fmt, ...);

#define RT_FAIL(code, fmt, ...) \
    return rt_err_set((code), 0, __FILE__, __LINE__, __func__, (fmt), ##__VA_ARGS__);

#define RT_FAIL_ERRNO(code, fmt, ...) \
    do { int _e = errno, return rt_err_set((code), _e, __FILE__, __LINE__, __func__, (fmt), ##__VA_ARGS__); }

#define RT_CHECK(expr) \
    do { int _rc = (expr); if (_rc < 0) return _rc; } while (0)

#define RT_CHECK_GOTO(expr, label) \
    do { int _rc = (expr); if (_rc < 0) goto label; } while (0)

#endif
