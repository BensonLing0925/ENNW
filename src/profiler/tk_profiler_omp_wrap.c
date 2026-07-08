#include "tk_profiler.h"
#include <omp.h>

void __real_GOMP_parallel(void (*fn)(void *), void *data,
                          unsigned nthreads, unsigned flags);
typedef struct {
    void                   (*real_fn)(void *);
    void                    *real_data;
    struct tk_prof_manager  *manager;
    size_t                   mem_offset;
    const char*				 label;
    int                      observed_threads;
} _OmpWrapCtx;

static void _probe_fn(void *arg) {
    _OmpWrapCtx *w = (_OmpWrapCtx *)arg;

    #pragma omp single nowait
    {
        w->observed_threads = omp_get_num_threads();
    }

	tk_prof_record(w->manager, TK_EV_OMP_BEGIN, w->label, w->mem_offset);
    w->real_fn(w->real_data);
    tk_prof_record(w->manager, TK_EV_OMP_END, w->label, w->mem_offset);
}

void __wrap_GOMP_parallel(void (*fn)(void *), void *data,
                          unsigned nthreads, unsigned flags) {
    struct tk_prof_manager *mgr = tk_prof_get_my_manager();
    if (!mgr) {
        __real_GOMP_parallel(fn, data, nthreads, flags);
        return;
    }

    _OmpWrapCtx ctx = {
        .real_fn           = fn,
        .real_data         = data,
        .manager           = mgr,
        .mem_offset        = tk_prof_get_offset(),
        .label             = tk_prof_get_label(),
        .observed_threads  = 0,
    };
    __real_GOMP_parallel(_probe_fn, &ctx, nthreads, flags);

    tk_prof_set_last_omp_threads(ctx.observed_threads);
}
