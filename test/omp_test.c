#include "tk_profiler.h"
#include "tensor.h"
#include "tensor_ops.h"
#include <omp.h>

void thread_cost_test() {
    uint64_t t0 = tk_get_now_ns();
    for (int i = 0; i < 1000; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < omp_get_max_threads(); ++j) {}
    }
    uint64_t t = tk_get_now_ns() - t0;
    printf("fork/join cost: %.3f us per region (%d threads)\n",
           t / 1000.0 / 1000, omp_get_max_threads());
}

void gemm_size_test() {

    struct arena a;
    arena_init(&a);

    int sizes[][3] = {
        {5, 5, 64},        /* attention score: [5,64] x [64,5] */
        {5, 64, 5},        /* attention out:   [5,5] x [5,64]  */
        {5, 768, 768},     /* QKV projection                   */
        {5, 3072, 768},    /* FFN up                           */
        {1, 768, 768},     /* decode QKV (M=1)                 */
        {1, 50257, 768},   /* LM head                          */
    };

    // tk_ops_gemm(struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest)
    /*
    int tk_tensor_alloc(struct arena* a,
                    enum tk_dtype dtype,
                    int* shape,
                    int ndims,
                    struct tk_tensor** out) {
    */

    for (int test = 0 ; test < 6 ; ++test) {
        int i = test;
        struct tk_tensor* src1;
        struct tk_tensor* src2;
        struct tk_tensor* dest;
        int m = sizes[i][0];
        int k = sizes[i][2];
        int n = sizes[i][1];
        tk_tensor_alloc(&a, TK_F32, (int[]){m, k}, 2, &src1);
        tk_tensor_alloc(&a, TK_F32, (int[]){k, n}, 2, &src2);
        tk_tensor_alloc(&a, TK_F32, (int[]){m, n}, 2, &dest);
        tk_tensor_rand_init(src1, 3.0);
        tk_tensor_rand_init(src2, 3.0);
        
        int reps = (m * n * k < 100000) ? 10000 : 100;
        uint64_t t0 = tk_get_now_ns();
        for (int r = 0; r < reps; ++r)
            tk_ops_gemm(src1, src2, dest);
        double us = (tk_get_now_ns() - t0) / 1000.0 / reps;
        printf("[%d,%d] x [%d,%d] : %.2f us\n", m, k, k, n, us);
    }
    arena_destroy(&a);
}

int main() {
    thread_cost_test();
    gemm_size_test();
}
