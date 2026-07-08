#include "../error/rt_error.h"
#include "tk_profiler.h"
#include <time.h>
#include <stdint.h>

static __thread struct tk_prof_thread_buf* my_buf = NULL;
static __thread struct tk_prof_manager*    my_manager = NULL;
static __thread size_t                     my_pending_offset = 0;
static __thread int                        my_last_omp_threads = 0;
static __thread const char* 			   my_pending_label = NULL;

#ifdef _WIN32
#include <windows.h>
#endif

const char* tk_ev_type_get(enum tk_event_type type) {
    switch (type) {
        case TK_EV_OP_BEGIN:
            return "TK_EV_OP_BEGIN";
        case TK_EV_OP_END:
            return "TK_EV_OP_END";
        case TK_EV_OMP_BEGIN:
            return "TK_EV_OMP_BEGIN";
        case TK_EV_OMP_END:
            return "TK_EV_OMP_END";
        case TK_EV_ALLOC:
            return "TK_EV_ALLOC";
        case TK_EV_FREE:
            return "TK_EV_FREE\n";
        default:
            printf("Unknown event type\n");
            return NULL;
    }	
}

/*
void event_print(struct tk_event* ev, size_t mem_diff) {
    if (ev->type == TK_EV_OP_END && mem_diff > 0) {
        printf("[%10.4f] [%-3s] [T%02d] %-15s [offs:0x%08llX] [\x1b[32m+%6zu KB\x1b[0m] : %s\n", 
            ev->timestamp / 1e9, "CPU", ev->thread_id, ev->label, 
            (unsigned long long)ev->mem_offset, mem_diff / 1024, tk_ev_type_get(ev->type));
    } else {
        printf("[%10.4f] [%-3s] [T%02d] %-15s [offs:0x%08llX] [          ] : %s\n", 
            ev->timestamp / 1e9, "CPU", ev->thread_id, ev->label, 
            (unsigned long long)ev->mem_offset, tk_ev_type_get(ev->type));
    }
}
*/

void event_print(struct tk_event* ev, size_t mem_diff) {
    const char* type_str = tk_ev_type_get(ev->type);

    if (ev->type == TK_EV_OP_END && mem_diff > 0 && ev->omp_threads > 0) {
        printf("[%10.4f] [%-3s] [T%02d] %-15s [offs:0x%08llX] [\x1b[32m+%6zu KB\x1b[0m] [T*%03d] : %s\n",
            ev->timestamp / 1e9, "CPU", ev->thread_id, ev->label,
            (unsigned long long)ev->mem_offset, mem_diff / 1024,
            ev->omp_threads, type_str);

    } else if (ev->type == TK_EV_OP_END && mem_diff > 0) {
        printf("[%10.4f] [%-3s] [T%02d] %-15s [offs:0x%08llX] [\x1b[32m+%6zu KB\x1b[0m] [     ] : %s\n",
            ev->timestamp / 1e9, "CPU", ev->thread_id, ev->label,
            (unsigned long long)ev->mem_offset, mem_diff / 1024, type_str);

    } else if (ev->type == TK_EV_OP_END && ev->omp_threads > 0) {
        printf("[%10.4f] [%-3s] [T%02d] %-15s [offs:0x%08llX] [          ] [T*%03d] : %s\n",
            ev->timestamp / 1e9, "CPU", ev->thread_id, ev->label,
            (unsigned long long)ev->mem_offset, ev->omp_threads, type_str);

    } else {
        printf("[%10.4f] [%-3s] [T%02d] %-15s [offs:0x%08llX] [          ] [     ] : %s\n",
            ev->timestamp / 1e9, "CPU", ev->thread_id, ev->label,
            (unsigned long long)ev->mem_offset, type_str);
    }
}

void thread_print(struct tk_prof_thread_buf* buf) {
    printf("Thread ID: %d\n", buf->thread_id);
    printf("Event count: %d\n", (int)buf->head);

    for (int i = 0; i < buf->head; ++i) {
        struct tk_event* ev = &buf->events[i];
        size_t mem_diff = 0;

        if (ev->type == TK_EV_OP_END) {
            for (int j = i - 1; j >= 0; --j) {
                if (buf->events[j].type == TK_EV_OP_BEGIN && 
                    strcmp(buf->events[j].label, ev->label) == 0) {
                    mem_diff = ev->mem_offset - buf->events[j].mem_offset;
                    break;
                }
            }
        }
        event_print(ev, mem_diff);
    }
}

void threads_print(struct tk_prof_manager* manager) {
	if (!manager) return;
	int num_threads = manager->thread_count;
	for ( int i = 0 ; i < num_threads ; ++i )
		thread_print(&manager->thread_pool[i]);
}

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


struct tk_prof_manager* tk_prof_manager_create(struct arena* prof_arena) {
    return arena_alloc(prof_arena, sizeof(struct tk_prof_manager));
}

// caller should init arena
// should be pre-allocated even before dryrun to avoid using mutex
struct tk_prof_manager* tk_prof_create(int max_threads,
                                       size_t max_events_per_thread) {

    // boostrap tk_prof_manager inside prof_arena
    ARENA_EMPTY(local_arena);
    struct tk_prof_manager* manager = tk_prof_manager_create(&local_arena);
    struct arena* prof_arena = arena_alloc(&local_arena, sizeof(struct arena));
    *prof_arena = local_arena;
    manager->prof_arena = prof_arena;

    manager->max_threads = max_threads;
    manager->thread_pool = arena_alloc(prof_arena, sizeof(struct tk_prof_thread_buf) * max_threads);
    manager->thread_count = 0;
    manager->global_start_ns = tk_get_now_ns();

    struct tk_prof_thread_buf* bufs = manager->thread_pool;

    for ( int i = 0 ; i < max_threads ; ++i ) {
        bufs[i].events = arena_alloc(prof_arena, sizeof(struct tk_event) * max_events_per_thread);
        bufs[i].capacity = max_events_per_thread;
        bufs[i].head = 0;
        bufs[i].thread_id = i;
        bufs[i].thread_name = NULL;
    }

    return manager;

}

int tk_prof_thread_attach(struct tk_prof_manager* manager) {
    int idx = atomic_fetch_add(&manager->thread_count, 1);
    if (idx >= manager->max_threads) {
        RT_FAIL(RT_EINTERNAL, __func__, "Max threads exceeded\n");
    }
    else {
        my_buf = &manager->thread_pool[idx];
        my_buf->thread_id = idx;
        my_manager = manager;
    }
    return 0;
}

void tk_prof_record(struct tk_prof_manager* manager, enum tk_event_type type, const char* label, size_t offset) {

    if (unlikely(!manager)) return;

    if (unlikely(!my_buf)) {
        if (tk_prof_thread_attach(manager) != 0) return;
    }

    int head = my_buf->head;

    if (likely(my_buf->head < my_buf->capacity)) {
        struct tk_event* ev = &my_buf->events[my_buf->head++];
        
        ev->timestamp = tk_get_now_ns() - manager->global_start_ns;
        ev->type = type;
        ev->label = label;
        ev->mem_offset = offset;
        ev->thread_id = my_buf->thread_id;
        ev->omp_threads = (type == TK_EV_OP_END) ? my_last_omp_threads : 0;
    }
}

// functions for implicit struct dereference and local threads
void tk_prof_emit(struct tk_rt_ctx* ctx, int type, const char* label, size_t offset) {
    if (ctx && ctx->use_prof) {
    	if (type == TK_EV_OP_BEGIN) {
            tk_prof_set_offset(offset);
            tk_prof_set_last_omp_threads(0);
            tk_prof_set_label(label);
        }
        tk_prof_record(ctx->manager, type, label, offset);
    }
}

struct tk_prof_manager* tk_prof_get_my_manager(void) {
    return my_manager;
}

void tk_prof_set_offset(size_t offset) {
    my_pending_offset = offset;
}

size_t tk_prof_get_offset(void) {
    return my_pending_offset;
}

void tk_prof_set_last_omp_threads(int n) {
    my_last_omp_threads = n;
}

int tk_prof_get_last_omp_threads(void) {
    return my_last_omp_threads;
}

void tk_prof_set_label(const char* label) {
    my_pending_label = label;
}
const char* tk_prof_get_label(void) {
    return my_pending_label;
}
