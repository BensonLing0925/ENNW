#include "tk_profiler_view.h"
#include "tk_profiler_span.h"
#include "raylib.h"
#include <stdio.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Layout
 * ----------------------------------------------------------------------- */
#define WIN_W           1400
#define WIN_H           760
#define STATS_W         320
#define TL_W            (WIN_W - STATS_W)
#define LABEL_W         152
#define HEADER_H        44
#define RULER_H         30
#define ROW_H           38
#define ROW_GAP         5
#define HEATMAP_H       52
#define HM_BUCKETS      600
#define FONT_SM         11
#define FONT_MD         13
#define FONT_LG         17
#define MAX_OP_SPANS    4096
#define MAX_OMP_SPANS   65536
#define OMP_GAP_NS      500000ULL
#define MAX_OP_TYPES    8
#define MAX_THREADS     64

/* -----------------------------------------------------------------------
 * Colors
 * ----------------------------------------------------------------------- */
#define C_BG        ((Color){13,  17,  23,  255})
#define C_PANEL     ((Color){20,  25,  33,  255})
#define C_PANEL2    ((Color){26,  31,  40,  255})
#define C_BORDER    ((Color){44,  51,  61,  255})
#define C_TEXT      ((Color){225, 232, 240, 255})
#define C_DIM       ((Color){120, 130, 145, 255})
#define C_GRID      ((Color){26,  31,  40,  200})
#define C_HOVER     ((Color){255, 255, 255,  12})
#define C_ACCENT    ((Color){ 88, 166, 255, 255})

static const Color OP_COLORS[MAX_OP_TYPES] = {
    { 88, 166, 255, 210},  /* ATTENTION       */
    {240, 136,  62, 210},  /* FFN             */
    { 60, 190,  90, 210},  /* FUSED_ADD_NORM  */
    {200, 110, 170, 210},  /* GEMM            */
    { 60, 160, 200, 210},  /* LAYERNORM       */
    {230, 210,  55, 210},  /* GELU            */
    {210,  80,  60, 210},  /* QUANTIZE        */
    {140, 140, 140, 210},  /* other           */
};

static const char* OP_ORDER[] = {
    "ATTENTION","FFN","FUSED_ADD_NORM","GEMM",
    "LAYERNORM","GELU","QUANTIZE", NULL
};

/* -----------------------------------------------------------------------
 * Data structures
 * ----------------------------------------------------------------------- */
typedef struct {
    char   label[32];
    Color  color;
    double total_ms, avg_ms;
    int    count;
    float  avg_threads, pct;
    int    expanded, row_y;
    int    hm[HM_BUCKETS];
    int    hm_max;
} OpInfo;

typedef struct {
    double t_min, range;
    float  zoom, scroll;
} VP;

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */
static int vp_px(const VP* v, uint64_t ts) {
    double r = ((double)ts - v->t_min) / v->range;
    return LABEL_W + (int)(r * (TL_W - LABEL_W - 6) * v->zoom - v->scroll);
}

static Color lerp_color(Color a, Color b, float t) {
    if (t < 0) t = 0; if (t > 1) t = 1;
    return (Color){
        (unsigned char)(a.r + (b.r - a.r) * t),
        (unsigned char)(a.g + (b.g - a.g) * t),
        (unsigned char)(a.b + (b.b - a.b) * t),
        (unsigned char)(a.a + (b.a - a.a) * t),
    };
}

static void stat_bar(int x, int y, int w, int h, float pct, Color fill) {
    DrawRectangle(x, y, w, h, (Color){28, 36, 48, 255});
    int fw = (int)(w * pct);
    if (fw > 0) { Color f = fill; f.a = 255; DrawRectangle(x, y, fw, h, f); }
}

static void build_heatmap(OpInfo* info, const tk_omp_span* sp, int n,
                           double t_min, double range) {
    memset(info->hm, 0, sizeof(info->hm));
    info->hm_max = 1;
    for (int i = 0; i < n; i++) {
        if (strcmp(sp[i].label, info->label) != 0) continue;
        int b0 = (int)(((double)sp[i].first_begin_ns - t_min) / range * HM_BUCKETS);
        int b1 = (int)(((double)sp[i].last_end_ns    - t_min) / range * HM_BUCKETS);
        if (b0 < 0) b0 = 0;
        if (b1 >= HM_BUCKETS) b1 = HM_BUCKETS - 1;
        for (int b = b0; b <= b1; b++) {
            info->hm[b]++;
            if (info->hm[b] > info->hm_max) info->hm_max = info->hm[b];
        }
    }
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */
void tk_prof_view_run(struct tk_prof_manager* manager) {

    static tk_prof_span op_spans[MAX_OP_SPANS];
    static tk_omp_span  omp_spans[MAX_OMP_SPANS];

    int n_op  = tk_prof_collect_spans(&manager->thread_pool[0],
                                      op_spans, MAX_OP_SPANS);
    int n_omp = tk_prof_collect_all_omp_spans(manager, omp_spans,
                                              MAX_OMP_SPANS, OMP_GAP_NS);
    if (n_op == 0) return;

    /* Time range */
    uint64_t t_min = (uint64_t)-1, t_max = 0;
    for (int i = 0; i < n_op; i++) {
        if (op_spans[i].start_ns < t_min) t_min = op_spans[i].start_ns;
        if (op_spans[i].end_ns   > t_max) t_max = op_spans[i].end_ns;
    }
    double total_ms = (t_max - t_min) / 1e6;

    /* Build op info */
    static OpInfo ops[MAX_OP_TYPES];
    int n_ops = 0;
    for (int k = 0; OP_ORDER[k] && n_ops < MAX_OP_TYPES; k++) {
        int found = 0;
        for (int i = 0; i < n_op && !found; i++)
            if (strcmp(op_spans[i].label, OP_ORDER[k]) == 0) found = 1;
        if (!found) continue;

        OpInfo* info   = &ops[n_ops];
        strncpy(info->label, OP_ORDER[k], 31);
        info->color       = OP_COLORS[k < MAX_OP_TYPES ? k : MAX_OP_TYPES - 1];
        info->total_ms    = 0; info->avg_threads = 0; info->count = 0;
        info->expanded    = 0;

        for (int i = 0; i < n_op; i++) {
            if (strcmp(op_spans[i].label, info->label) != 0) continue;
            info->total_ms    += (op_spans[i].end_ns - op_spans[i].start_ns) / 1e6;
            info->avg_threads += op_spans[i].omp_threads;
            info->count++;
        }
        if (info->count > 0) {
            info->avg_ms      = info->total_ms / info->count;
            info->avg_threads /= info->count;
        }
        info->pct = (float)(info->total_ms / total_ms * 100.0);
        build_heatmap(info, omp_spans, n_omp,
                      (double)t_min, (double)(t_max - t_min));
        n_ops++;
    }

    /* Per-thread active time */
    static double th_ms[MAX_THREADS];
    memset(th_ms, 0, sizeof(th_ms));
    for (int i = 0; i < n_omp; i++) {
        int tid = omp_spans[i].thread_id;
        if (tid > 0 && tid < MAX_THREADS)
            th_ms[tid] += (omp_spans[i].last_end_ns - omp_spans[i].first_begin_ns) / 1e6;
    }
    double th_max = 0;
    int n_workers = manager->thread_count - 1;
    for (int t = 1; t <= n_workers && t < MAX_THREADS; t++)
        if (th_ms[t] > th_max) th_max = th_ms[t];

    /* Viewport */
    VP vp = { (double)t_min, (double)(t_max - t_min), 1.0f, 0.0f };
    int tl_uw = TL_W - LABEL_W - 6;   /* usable timeline width */

    InitWindow(WIN_W, WIN_H, "TK Profiler View");
    SetTargetFPS(60);

    char tooltip[256] = {0};

    while (!WindowShouldClose()) {

        /* Input */
        Vector2 mouse = GetMousePosition();
        float wheel = GetMouseWheelMove();

        if (wheel != 0.0f && mouse.x < TL_W) {
            float r = ((float)(mouse.x - LABEL_W) + vp.scroll) / (tl_uw * vp.zoom);
            vp.zoom += wheel * 0.12f * vp.zoom;
            if (vp.zoom < 0.05f)   vp.zoom = 0.05f;
            if (vp.zoom > 1000.0f) vp.zoom = 1000.0f;
            vp.scroll = r * tl_uw * vp.zoom - (float)(mouse.x - LABEL_W);
        }
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && mouse.x < TL_W - 10)
            vp.scroll -= GetMouseDelta().x;
        float max_s = tl_uw * vp.zoom - tl_uw + 20;
        if (vp.scroll < 0) vp.scroll = 0;
        if (max_s > 0 && vp.scroll > max_s) vp.scroll = max_s;

        /* Row Y positions */
        int yc = HEADER_H + 10;
        for (int k = 0; k < n_ops; k++) {
            ops[k].row_y = yc;
            yc += ROW_H + ROW_GAP;
            if (ops[k].expanded) yc += HEATMAP_H + ROW_GAP;
        }

        BeginDrawing();
        ClearBackground(C_BG);
        tooltip[0] = 0;

        /* ===== TIMELINE PANEL ===== */

        /* Header */
        DrawRectangle(0, 0, TL_W, HEADER_H, C_PANEL);
        DrawLine(0, HEADER_H, TL_W, HEADER_H, C_BORDER);
        DrawText("TK PROFILER", 14, 13, FONT_LG, C_TEXT);
        char hdr[160];
        snprintf(hdr, sizeof(hdr),
                 "%.2f ms  ¡P  %d ops  ¡P  %d threads  ¡P  scroll=zoom  drag=pan  [+]=threads",
                 total_ms, n_op, manager->thread_count);
        DrawText(hdr, 158, 15, FONT_SM, C_DIM);

        /* Vertical grid */
        for (int t = 0; t <= 8; t++) {
            int gx = vp_px(&vp, (uint64_t)(t_min + vp.range * t / 8));
            if (gx >= LABEL_W && gx < TL_W)
                DrawLine(gx, HEADER_H, gx, WIN_H - RULER_H, C_GRID);
        }

        /* Label column */
        DrawRectangle(0, HEADER_H, LABEL_W, WIN_H - HEADER_H, C_PANEL2);
        DrawLine(LABEL_W, HEADER_H, LABEL_W, WIN_H, C_BORDER);

        /* Op rows */
        for (int k = 0; k < n_ops; k++) {
            OpInfo* info = &ops[k];
            int ry = info->row_y;

            if (mouse.y >= ry && mouse.y < ry + ROW_H && mouse.x < TL_W)
                DrawRectangle(LABEL_W, ry, TL_W - LABEL_W, ROW_H, C_HOVER);

            /* Color swatch */
            Color dot = info->color; dot.a = 255;
            DrawRectangle(10, ry + 13, 7, 13, dot);

            /* Label (truncate if needed) */
            char lbl[32]; strncpy(lbl, info->label, 31);
            while (strlen(lbl) > 1 && MeasureText(lbl, FONT_MD) > LABEL_W - 42)
                lbl[strlen(lbl)-1] = 0;
            DrawText(lbl, 23, ry + 12, FONT_MD, C_TEXT);

            /* Expand button */
            DrawRectangle(LABEL_W - 24, ry + 10, 18, 18, (Color){34, 42, 54, 255});
            DrawText(info->expanded ? "-" : "+", LABEL_W - 19, ry + 12, FONT_MD, C_DIM);
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                Rectangle btn = {LABEL_W - 26, ry + 8, 22, 22};
                if (CheckCollisionPointRec(mouse, btn))
                    info->expanded = !info->expanded;
            }

            /* Op span bars */
            for (int i = 0; i < n_op; i++) {
                if (strcmp(op_spans[i].label, info->label) != 0) continue;
                int px0 = vp_px(&vp, op_spans[i].start_ns);
                int px1 = vp_px(&vp, op_spans[i].end_ns);
                int pw = px1 - px0; if (pw < 1) pw = 1;
                if (px0 + pw < LABEL_W || px0 > TL_W) continue;
                if (px0 < LABEL_W) { pw -= LABEL_W - px0; px0 = LABEL_W; }

                DrawRectangle(px0, ry + 6, pw, ROW_H - 12, info->color);

                if (CheckCollisionPointRec(mouse, (Rectangle){px0, ry+6, pw, ROW_H-12})) {
                    size_t mk = op_spans[i].mem_after > op_spans[i].mem_before
                                ? (op_spans[i].mem_after - op_spans[i].mem_before) / 1024 : 0;
                    snprintf(tooltip, sizeof(tooltip),
                             "%s   %.3f ms   T*%d   +%zu KB",
                             info->label,
                             (op_spans[i].end_ns - op_spans[i].start_ns) / 1e6,
                             op_spans[i].omp_threads, mk);
                }
            }

            /* Thread heatmap */
            if (info->expanded) {
                int hy = ry + ROW_H + ROW_GAP;
                DrawRectangle(0,       hy, LABEL_W, HEATMAP_H, C_PANEL2);
                DrawRectangle(LABEL_W, hy, TL_W - LABEL_W, HEATMAP_H,
                              (Color){14, 19, 28, 255});
                DrawText("density", 8, hy + 8,  FONT_SM, C_DIM);
                char ml[16]; snprintf(ml, sizeof(ml), "T*%d", info->hm_max);
                DrawText(ml, 8, hy + HEATMAP_H - 16, FONT_SM, C_DIM);

                float bw_f = (float)tl_uw * vp.zoom / HM_BUCKETS;
                Color c_dark = {14, 19, 28, 255};
                Color c_full = info->color; c_full.a = 240;

                for (int b = 0; b < HM_BUCKETS; b++) {
                    if (info->hm[b] == 0) continue;
                    int bx = LABEL_W + (int)((float)b / HM_BUCKETS * tl_uw * vp.zoom - vp.scroll);
                    int bw = (int)bw_f + 1;
                    if (bx + bw < LABEL_W || bx > TL_W) continue;
                    if (bx < LABEL_W) { bw -= LABEL_W - bx; bx = LABEL_W; }
                    Color hc = lerp_color(c_dark, c_full,
                                          (float)info->hm[b] / info->hm_max);
                    DrawRectangle(bx, hy, bw, HEATMAP_H, hc);
                }
                DrawLine(LABEL_W, hy + HEATMAP_H, TL_W, hy + HEATMAP_H, C_BORDER);
            }

            DrawLine(0, ry + ROW_H + ROW_GAP / 2,
                     TL_W, ry + ROW_H + ROW_GAP / 2, C_BORDER);
        }

        /* Ruler */
        int ruler_y = WIN_H - RULER_H;
        DrawRectangle(0, ruler_y, TL_W, RULER_H, C_PANEL);
        DrawLine(0, ruler_y, TL_W, ruler_y, C_BORDER);
        for (int t = 0; t <= 8; t++) {
            int tx = vp_px(&vp, (uint64_t)(t_min + vp.range * t / 8));
            if (tx < LABEL_W || tx > TL_W - 4) continue;
            DrawLine(tx, ruler_y, tx, ruler_y + 5, C_DIM);
            char lbl[24];
            snprintf(lbl, sizeof(lbl), "%.1fms",
                     (double)(t_min + (uint64_t)(vp.range * t / 8) - t_min) / 1e6);
            DrawText(lbl, tx - 14, ruler_y + 8, FONT_SM, C_DIM);
        }

        /* ===== STATS PANEL ===== */
        int sx = TL_W, sw = STATS_W;
        DrawRectangle(sx, 0, sw, WIN_H, C_PANEL);
        DrawLine(sx, 0, sx, WIN_H, C_BORDER);

        /* Header */
        DrawRectangle(sx, 0, sw, HEADER_H, C_PANEL2);
        DrawLine(sx, HEADER_H, sx + sw, HEADER_H, C_BORDER);
        DrawText("SUMMARY", sx + 14, 13, FONT_LG, C_TEXT);

        int sy = HEADER_H + 16;

        /* Total time */
        DrawText("Total inference time", sx + 14, sy, FONT_SM, C_DIM); sy += 17;
        char ts[32]; snprintf(ts, sizeof(ts), "%.2f ms", total_ms);
        DrawText(ts, sx + 14, sy, FONT_LG + 6, C_TEXT); sy += 36;

        DrawLine(sx + 10, sy, sx + sw - 10, sy, C_BORDER); sy += 14;
        DrawText("TIME BREAKDOWN", sx + 14, sy, FONT_SM, C_DIM); sy += 18;

        int bw = sw - 28;
        for (int k = 0; k < n_ops; k++) {
            OpInfo* info = &ops[k];

            /* Label + pct */
            Color dot2 = info->color; dot2.a = 255;
            DrawRectangle(sx + 14, sy + 1, 7, 13, dot2);
            DrawText(info->label, sx + 26, sy, FONT_SM, C_TEXT);
            char pct[12]; snprintf(pct, sizeof(pct), "%.1f%%", info->pct);
            DrawText(pct, sx + sw - MeasureText(pct, FONT_SM) - 12,
                     sy, FONT_SM, C_DIM);
            sy += 16;

            /* Bar */
            stat_bar(sx + 14, sy, bw, 7, info->pct / 100.0f, info->color);
            sy += 11;

            /* Sub-stats */
            char sub[64];
            if (info->avg_threads > 0.5f)
                snprintf(sub, sizeof(sub), "%d calls  avg %.1fms  T*%.0f",
                         info->count, info->avg_ms, info->avg_threads);
            else
                snprintf(sub, sizeof(sub), "%d calls  avg %.1fms",
                         info->count, info->avg_ms);
            DrawText(sub, sx + 14, sy, FONT_SM - 1, C_DIM);
            sy += 20;
        }

        DrawLine(sx + 10, sy, sx + sw - 10, sy, C_BORDER); sy += 14;
        DrawText("THREAD UTILIZATION", sx + 14, sy, FONT_SM, C_DIM); sy += 18;

        if (n_workers > 0 && th_max > 0) {
            int th_h = 5, th_gap = 3;
            int avail = WIN_H - sy - 38;
            int show = avail / (th_h + th_gap);
            if (show > n_workers) show = n_workers;

            for (int t = 1; t <= show && t < MAX_THREADS; t++) {
                float util = (float)(th_ms[t] / th_max);
                Color tc = lerp_color((Color){32, 42, 58, 255}, C_ACCENT, util);
                stat_bar(sx + 14, sy, bw, th_h, util, tc);
                if (t == 1 || t == show || t % 5 == 0) {
                    char tid[8]; snprintf(tid, sizeof(tid), "T%02d", t);
                    DrawText(tid, sx + 14 + (int)(bw * util) + 4,
                             sy - 1, FONT_SM - 2, C_DIM);
                }
                sy += th_h + th_gap;
            }
            if (n_workers > show) {
                char more[32];
                snprintf(more, sizeof(more), "+ %d threads", n_workers - show);
                DrawText(more, sx + 14, sy + 4, FONT_SM - 1, C_DIM);
            }
        }

        DrawText("[ + ]  expand thread density",
                 sx + 14, WIN_H - 34, FONT_SM - 1, C_DIM);
        DrawText("scroll: zoom  ¡P  drag: pan",
                 sx + 14, WIN_H - 18, FONT_SM - 1, C_DIM);

        /* Tooltip */
        if (tooltip[0]) {
            int tw = MeasureText(tooltip, FONT_MD) + 22;
            int tx = (int)mouse.x + 16;
            int ty = (int)mouse.y - 36;
            if (tx + tw > WIN_W - 4) tx = WIN_W - tw - 4;
            if (ty < 4) ty = 4;
            DrawRectangle(tx - 8, ty - 6, tw, FONT_MD + 16, (Color){8, 12, 18, 235});
            DrawLine(tx - 8, ty - 6, tx - 8 + tw, ty - 6,
                     (Color){88, 166, 255, 160});
            DrawText(tooltip, tx, ty, FONT_MD, C_TEXT);
        }

        DrawFPS(TL_W - 56, WIN_H - 18);
        EndDrawing();
    }

    CloseWindow();
}
