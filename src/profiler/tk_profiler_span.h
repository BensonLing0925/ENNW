#ifndef TK_PROF_SPAN_H
#define TK_PROF_SPAN_H

#include <stdint.h>
#include "tk_profiler.h"

/* -----------------------------------------------------------------------
 * tk_prof_span  —  master thread 算子級別 span（OP_BEGIN / OP_END 配對）
 * ----------------------------------------------------------------------- */
typedef struct {
    const char* label;
    uint64_t    start_ns;
    uint64_t    end_ns;
    size_t      mem_before;
    size_t      mem_after;
    int         thread_id;
    int         omp_threads;   /* 實際啟動的 OMP 執行緒數，無 OMP 則為 0 */
} tk_prof_span;

/* -----------------------------------------------------------------------
 * tk_omp_span  —  worker thread 視角：同一算子的所有細粒 OMP region 合併
 * ----------------------------------------------------------------------- */
typedef struct {
    const char* label;
    uint64_t    first_begin_ns;
    uint64_t    last_end_ns;
    int         region_count;  /* 內部細粒 parallel region 數量 */
    int         thread_id;
} tk_omp_span;

/* -----------------------------------------------------------------------
 * API
 * ----------------------------------------------------------------------- */

/*
 * 從 master thread buffer（通常是 thread_pool[0]）收集算子級別 span。
 * 回傳實際寫入 out 的數量。
 */
int tk_prof_collect_spans(struct tk_prof_thread_buf* buf,
                          tk_prof_span* out, int max_spans);

/*
 * 從 worker thread buffer 收集並合併同 label 連續的 OMP_BEGIN/END 事件。
 * 相鄰 region 之間間隔小於 gap_threshold_ns 時視為同一批次合併。
 * 建議 gap_threshold_ns = 500000 (0.5ms)。
 * 回傳實際寫入 out 的數量。
 */
int tk_prof_collect_omp_spans(struct tk_prof_thread_buf* buf,
                               tk_omp_span* out, int max_spans,
                               uint64_t gap_threshold_ns);

/*
 * 一次收集所有 thread 的 omp_span，結果依 thread_id 排列。
 * out 需要足夠大：建議 max_threads * max_spans_per_thread。
 * 回傳總數。
 */
int tk_prof_collect_all_omp_spans(struct tk_prof_manager* manager,
                                   tk_omp_span* out, int max_total,
                                   uint64_t gap_threshold_ns);

#endif /* TK_PROF_SPAN_H */
