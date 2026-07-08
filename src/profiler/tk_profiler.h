#ifndef TK_PROFILER_H
#define TK_PROFILER_H

#include <stdatomic.h>
#include "../../mem/arena.h"
#include "../runtime/rt_context.h"

#define TK_MAX_THREAD 32

#if defined(__GNUC__) || defined(__clang__)
    #define likely(x)       __builtin_expect(!!(x), 1)
    #define unlikely(x)     __builtin_expect(!!(x), 0)
#else
    #define likely(x)       (x)
    #define unlikely(x)     (x)
#endif

enum tk_event_type {
    TK_EV_OP_BEGIN,
    TK_EV_OP_END,
    TK_EV_OMP_BEGIN,
    TK_EV_OMP_END,
    TK_EV_ALLOC,
    TK_EV_FREE
};

struct tk_rt_ctx;
extern int tk_rt_is_prof_enabled(struct tk_rt_ctx* ctx);

struct tk_event {
    uint64_t timestamp;
    enum tk_event_type type;
    const char* label;
    size_t mem_offset;
    int thread_id;      // reserved for multi-thread
    int omp_threads;
};

struct tk_prof_thread_buf {
    struct tk_event* events;
    int capacity;
    int head;
    int thread_id;
    const char* thread_name;
};

struct tk_prof_manager {
    struct tk_prof_thread_buf* thread_pool;
    struct arena* prof_arena;
    _Atomic int thread_count; 
    uint64_t global_start_ns; 
    int max_threads;
};

#ifdef PROF
    #define TK_ENABLE_PROFILER
#endif

// control interface using macro
#ifdef TK_ENABLE_PROFILER
    
    void tk_prof_init(struct arena* prof_arena, size_t max_events);
    void tk_prof_shutdown();
    void tk_prof_record(struct tk_prof_manager* manager, enum tk_event_type type, const char* label, size_t offset);
    int tk_prof_thread_attach(struct tk_prof_manager* manager);
    struct tk_prof_manager* tk_prof_create(int max_threads, size_t max_events_per_thread);

    void tk_prof_emit(struct tk_rt_ctx* ctx, int type, const char* label, size_t offset);
    // void event_print(struct tk_event* ev);
    void event_print(struct tk_event* ev, size_t mem_diff);
    void thread_print(struct tk_prof_thread_buf* buf);
    struct tk_prof_manager* tk_prof_get_my_manager(void);
    void   tk_prof_set_offset(size_t offset);
	size_t tk_prof_get_offset(void);
	void tk_prof_set_last_omp_threads(int n);
	int  tk_prof_get_last_omp_threads(void);
    #define TK_PROF_SCOPE(ctx, type, label, offset) \
        do { \
            tk_prof_emit(ctx, type, label, offset); \
        } while(0)


#else
    static inline void tk_prof_init(struct arena* prof_arena, size_t max_events) {}
    static inline void tk_prof_shutdown() {}
    static inline void tk_prof_record(struct tk_prof_manager* manager, enum tk_event_type type, const char* label, size_t offset) {}
    static inline int tk_prof_thread_attach(struct tk_prof_manager* manager) {return 0;}
    struct tk_prof_manager* tk_prof_create(int max_threads, size_t max_events_per_thread) {return NULL;}
    void tk_prof_emit(struct tk_rt_ctx* ctx, int type, const char* label, size_t offset) {}
    // void event_print(struct tk_event* ev) {}
    void event_print(struct tk_event* ev, size_t mem_diff) {}
    void thread_print(struct tk_prof_thread_buf* buf) {}
    struct tk_prof_manager* tk_prof_get_my_manager(void) {}
    void   tk_prof_set_offset(size_t offset) {}
	size_t tk_prof_get_offset(void) {}
	void tk_prof_set_last_omp_threads(int n) {}
	int  tk_prof_get_last_omp_threads(void) {}
    #define TK_PROF_SCOPE(ctx, type, label, offset) do {} while(0)
#endif

#endif
