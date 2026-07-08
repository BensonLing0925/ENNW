#include "tk_profiler_span.h"
#include <string.h>

/* -----------------------------------------------------------------------
 * tk_prof_collect_spans
 * 掃描 master thread buffer，配對 OP_BEGIN / OP_END，輸出算子級 span。
 * ----------------------------------------------------------------------- */
int tk_prof_collect_spans(struct tk_prof_thread_buf* buf,
                          tk_prof_span* out, int max_spans) {
    int count = 0;

    for (int i = 0; i < buf->head && count < max_spans; i++) {
        struct tk_event* begin = &buf->events[i];
        if (begin->type != TK_EV_OP_BEGIN) continue;

        /* 找對應的 OP_END（同 label，往後第一個） */
        for (int j = i + 1; j < buf->head; j++) {
            struct tk_event* end = &buf->events[j];
            if (end->type != TK_EV_OP_END) continue;
            if (strcmp(end->label, begin->label) != 0) continue;

            tk_prof_span* s  = &out[count++];
            s->label         = begin->label;
            s->start_ns      = begin->timestamp;
            s->end_ns        = end->timestamp;
            s->mem_before    = begin->mem_offset;
            s->mem_after     = end->mem_offset;
            s->thread_id     = begin->thread_id;
            s->omp_threads   = end->omp_threads;
            break;
        }
    }

    return count;
}

/* -----------------------------------------------------------------------
 * tk_prof_collect_omp_spans
 * 掃描 worker thread buffer，把相鄰且同 label 的 OMP_BEGIN/END 合併。
 * ----------------------------------------------------------------------- */
int tk_prof_collect_omp_spans(struct tk_prof_thread_buf* buf,
                               tk_omp_span* out, int max_spans,
                               uint64_t gap_threshold_ns) {
    int count = 0;
    int i     = 0;

    while (i < buf->head && count < max_spans) {
        struct tk_event* ev = &buf->events[i];

        if (ev->type != TK_EV_OMP_BEGIN) { i++; continue; }

        /* 開始一個新的合併 span */
        tk_omp_span* s    = &out[count];
        s->label          = ev->label;
        s->first_begin_ns = ev->timestamp;
        s->last_end_ns    = ev->timestamp;
        s->region_count   = 0;
        s->thread_id      = ev->thread_id;

        uint64_t last_end = ev->timestamp;

        while (i < buf->head) {
            struct tk_event* cur = &buf->events[i];

            /* label 換了 → 進入下一個算子，結束本次合併 */
            if (strcmp(cur->label, s->label) != 0) break;

            /* 遇到 BEGIN：檢查是否和上一個 END 距離太遠 */
            if (cur->type == TK_EV_OMP_BEGIN) {
                if (s->region_count > 0 &&
                    cur->timestamp - last_end > gap_threshold_ns) {
                    /* 間隔太大，視為不同批次，結束本次合併 */
                    break;
                }
                s->region_count++;
            } else if (cur->type == TK_EV_OMP_END) {
                last_end       = cur->timestamp;
                s->last_end_ns = cur->timestamp;
            }

            i++;
        }

        /* 至少有一個 region 才輸出 */
        if (s->region_count > 0) count++;
    }

    return count;
}

/* -----------------------------------------------------------------------
 * tk_prof_collect_all_omp_spans
 * 遍歷所有 thread buffer，依序收集並合併。
 * ----------------------------------------------------------------------- */
int tk_prof_collect_all_omp_spans(struct tk_prof_manager* manager,
                                   tk_omp_span* out, int max_total,
                                   uint64_t gap_threshold_ns) {
    int total      = 0;
    int n_threads  = manager->thread_count;

    for (int t = 0; t < n_threads && total < max_total; t++) {
        struct tk_prof_thread_buf* buf = &manager->thread_pool[t];

        /* 跳過 master thread（T00 的 OMP 事件由 master 觀測，不在 worker buffer） */
        if (buf->thread_id == 0) continue;

        int got = tk_prof_collect_omp_spans(buf,
                                            out + total,
                                            max_total - total,
                                            gap_threshold_ns);
        total += got;
    }

    return total;
}
