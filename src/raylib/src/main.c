#include "raylib.h"
#include <stdio.h>

// 模擬算子節點結構
typedef struct {
    Rectangle rect;
    Color color;
    const char* name;
    bool isOptimized;
} OpNode;

int main(void) {
    // 1. 初始化視窗
    const int screenWidth = 800;
    const int screenHeight = 450;
    InitWindow(screenWidth, screenHeight, "Raylib - Transformer Op Test");

    // 2. 定義一個算子節點 (例如 MatMul)
    OpNode matmulNode = {
        .rect = { screenWidth/2 - 60, screenHeight/2 - 25, 120, 50 },
        .color = BLUE,
        .name = "MatMul (Q*K)",
        .isOptimized = false
    };

    SetTargetFPS(60);

    // 3. 主要遊戲迴圈
    while (!WindowShouldClose()) {
        // --- 邏輯更新 ---
        Vector2 mousePoint = GetMousePosition();
        
        // 互動：點擊節點切換「優化狀態」
        if (CheckCollisionPointRec(mousePoint, matmulNode.rect)) {
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                matmulNode.isOptimized = !matmulNode.isOptimized;
                matmulNode.color = matmulNode.isOptimized ? GREEN : BLUE;
            }
        }

        // --- 繪圖層級 ---
        BeginDrawing();
            ClearBackground(RAYWHITE);

            DrawText("Transformer Operator Visualization Test", 20, 20, 20, DARKGRAY);
            DrawText("Click the node to toggle 'Optimization' status", 20, 50, 16, GRAY);

            // 繪製連線（模擬輸入/輸出線）
            DrawLine(matmulNode.rect.x + 60, 0, matmulNode.rect.x + 60, matmulNode.rect.y, GRAY);
            DrawLine(matmulNode.rect.x + 60, matmulNode.rect.y + 50, matmulNode.rect.x + 60, screenHeight, GRAY);

            // 繪製節點本體
            DrawRectangleRec(matmulNode.rect, matmulNode.color);
            DrawRectangleLinesEx(matmulNode.rect, 2, DARKBLUE);

            // 繪製算子名稱
            int textWidth = MeasureText(matmulNode.name, 10);
            DrawText(matmulNode.name, matmulNode.rect.x + (120 - textWidth)/2, matmulNode.rect.y + 20, 10, WHITE);

            if (matmulNode.isOptimized) {
                DrawText("STATUS: OPTIMIZED", matmulNode.rect.x, matmulNode.rect.y + 60, 10, DARKGREEN);
            }

        EndDrawing();
    }

    // 4. 關閉資源
    CloseWindow();

    return 0;
}
