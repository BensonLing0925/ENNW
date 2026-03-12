#include "rt_error.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define RT_THREAD_LOCAL _Thread_local
#else
#define RT_THREAD_LOCAL
#endif

static RT_THREAD_LOCAL struct rt_err_status g_rt_err_status;

void err_status_clear(void) {
    memset(&g_rt_err_status, 0, sizeof(g_rt_err_status));
}

const struct rt_err_status* rt_err_last(void) {
    return &g_rt_err_status;
}

const char* rt_errc_str(rt_errc c) {
    switch (c) {
        case RT_OK: return "RT_OK";
        case RT_EINVAL: return "RT_EINVAL";
        case RT_EOOM: return "RT_EOOM";
        case RT_EIO: return "RT_EIO";
        case RT_ESTATE: return "RT_ESTATE";
        case RT_EINTERNAL: return "RT_EINTERNAL";
        default: return "RT_EUNKNOWN";
    }
}


int rt_err_set(rt_errc code, int sys_errno,
               const char* file, int line, const char* func,
               const char* fmt, ...) {
    
    g_rt_err_status.code = code;
    g_rt_err_status.sys_errno = sys_errno;
    g_rt_err_status.file = file;
    g_rt_err_status.line = line;
    g_rt_err_status.func = func;
    
    g_rt_err_status.msg[0] = '\0';
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        // add "[ERROR] at the beginning
        memcpy(g_rt_err_status.msg, "[ERROR] ", 8);
        vsnprintf(g_rt_err_status.msg, sizeof(g_rt_err_status.msg), fmt, ap);
        va_end(ap);
    }

    return -(int)code;

}

void rt_err_print(FILE* out) {
    if (!out) out = stderr;
    const struct rt_err_status* e = rt_err_last();
    if (!e || e->code == RT_OK) {
        fprintf(out, "[RT] OK\n");
        return;
    }

    fprintf(out, "[RT][ERROR] %s (%d)\n", rt_errc_str(e->code), (int)e->code);
    if (e->msg[0]) fprintf(out, "  msg  : %s\n", e->msg);
    if (e->file)   fprintf(out, "  where: %s:%d (%s)\n", e->file, e->line, e->func ? e->func : "?");
    if (e->sys_errno)
        fprintf(out, "  errno: %d (%s)\n", e->sys_errno, strerror(e->sys_errno));
}
