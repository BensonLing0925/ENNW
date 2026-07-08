#ifndef TK_PROF_VIEW_H
#define TK_PROF_VIEW_H

#include "tk_profiler.h"

/* -----------------------------------------------------------------------
 * tk_prof_view  —  Raylib 視覺化介面
 *
 * 依賴：
 *   - raylib（需自行在 Makefile 加 -lraylib）
 *   - tk_prof_span.h / tk_prof_span.c
 *
 * 編譯範例：
 *   gcc ... tk_prof_span.c tk_prof_view.c -lraylib -lm -fopenmp -o nn.exe
 *
 * 使用：
 *   推理結束後呼叫 tk_prof_view_run(manager)，視窗關閉後程式繼續。
 * ----------------------------------------------------------------------- */

/*
 * 開啟 Raylib 視窗，顯示：
 *   Layer 0：master thread 算子級別 Gantt（OP_BEGIN/END span）
 *   Layer 1：每條 worker thread 的 OMP span（合併後）
 *
 * 操作：
 *   滑鼠滾輪   — 縮放時間軸
 *   左鍵拖曳   — 平移
 *   滑鼠懸停   — tooltip 顯示詳細資訊
 *   ESC / 關閉 — 離開
 */
void tk_prof_view_run(struct tk_prof_manager* manager);

#endif /* TK_PROF_VIEW_H */
