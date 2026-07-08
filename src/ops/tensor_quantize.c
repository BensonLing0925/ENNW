#include <omp.h>
#include <stdint.h>
#include "../error/rt_error.h"
#include "float.h"
#include "math.h"
#include "tensor.h"
#include "../modules/pooling/pooling.h"
#include "tensor_quantize.h"

/* input: a list of integer
 * output: a list of corresponding vectors
 * Caller pre-allocate output->data 
 */
/* ---- int8 dynamic quantization ---- */

float tk_ops_dyn_quantize_f32_i8(struct tk_tensor* src_f32, struct tk_tensor* dst_i8) {
    if (src_f32->dtype != TK_F32 || dst_i8->dtype != TK_I8)
        return 0.0f;

    uint64_t n = shape_size_calc(src_f32->shape, src_f32->ndims);
    const float* s = (const float*)src_f32->data;
    int8_t*      d = (int8_t*)dst_i8->data;

    float max_val = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        float v = fabsf(s[i]);
        if (v > max_val) max_val = v;
    }

    float scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
    float inv   = 1.0f / scale;

    for (uint64_t i = 0; i < n; i++) {
        float  q  = s[i] * inv;
        int32_t qi = (q >= 0.0f) ? (int32_t)(q + 0.5f) : (int32_t)(q - 0.5f);
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        d[i] = (int8_t)qi;
    }

    return scale;
}

void tk_ops_static_quantize_f32_i8(struct tk_tensor* src_f32,
                                    struct tk_tensor* dst_i8,
                                    float scale) {
    uint64_t n = shape_size_calc(src_f32->shape, src_f32->ndims);
    const float* s = (const float*)src_f32->data;
    int8_t*      d = (int8_t*)dst_i8->data;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
    for (uint64_t i = 0; i < n; i++) {
        float  q  = s[i] * inv;
        int32_t qi = (q >= 0.0f) ? (int32_t)(q + 0.5f) : (int32_t)(q - 0.5f);
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        d[i] = (int8_t)qi;
    }
}

/* int8 x int8 GEMM -> float32
 * Accumulates products in int32 to avoid overflow (max 127*127*4096 < INT32_MAX).
 * Dequantizes the final result: out_f32 = acc_i32 * act_scale * weight_scale
 */
int tk_ops_gemm_i8f32(struct tk_tensor* src1, struct tk_tensor* src2,
                      struct tk_tensor* dst) {
    if (src1->dtype != TK_I8 || src2->dtype != TK_I8)
        RT_FAIL(RT_EINVAL, "tk_ops_gemm_i8f32: src tensors must be TK_I8\n");
    if (dst->dtype != TK_F32)
        RT_FAIL(RT_EINVAL, "tk_ops_gemm_i8f32: dst tensor must be TK_F32\n");

    int err = batch_shape_equal_check(src1, src2);
    if (err) return err;

    float act_scale = src1->scale;
    float weight_scale = src2->scale;

    int p  = src1->shape[src1->ndims - 2];
    int q  = src1->shape[src1->ndims - 1];
    int q2 = src2->shape[src2->ndims - 2];
    int r  = src2->shape[src2->ndims - 1];

    if (q != q2)
        RT_FAIL(RT_EINVAL, "gemm_i8f32: inner dim mismatch %d vs %d\n", q, q2);
    if (dst->shape[dst->ndims-2] != p || dst->shape[dst->ndims-1] != r)
        RT_FAIL(RT_EINVAL, "gemm_i8f32: dst shape mismatch\n");

    uint64_t num_batches = 1;
    for (int i = 0; i < src1->ndims - 2; i++) num_batches *= (uint64_t)src1->shape[i];

    uint64_t bs1 = (num_batches > 1) ? (uint64_t)p * q : 0;
    uint64_t bs2 = (num_batches > 1) ? (uint64_t)q * r : 0;
    uint64_t bsd = (uint64_t)p * r;

    float dequant = act_scale * weight_scale;

    const int8_t* A_base = (const int8_t*)src1->data;
    const int8_t* B_base = (const int8_t*)src2->data;
    float*        C_base = (float*)dst->data;

    const int tile_k = 32;
    const int tile_j = 32;

    for (uint64_t b = 0; b < num_batches; b++) {
        const int8_t* A = A_base + b * bs1;
        const int8_t* B = B_base + b * bs2;
        float*        C = C_base + b * bsd;

        _Pragma("omp parallel for schedule(static)")
        for (int i = 0; i < p; i++) {
            /* per-row int32 accumulation buffer on the thread stack */
            int32_t* acc = (int32_t*)__builtin_alloca((size_t)r * sizeof(int32_t));
            memset(acc, 0, (size_t)r * sizeof(int32_t));

            for (int kk = 0; kk < q; kk += tile_k) {
                int k_lim = (kk + tile_k < q) ? kk + tile_k : q;
                for (int jj = 0; jj < r; jj += tile_j) {
                    int j_lim = (jj + tile_j < r) ? jj + tile_j : r;
                    for (int k = kk; k < k_lim; k++) {
                        int32_t a = (int32_t)A[(uint64_t)i * q + k];
                        for (int j = jj; j < j_lim; j++) {
                            acc[j] += a * (int32_t)B[(uint64_t)k * r + j];
                        }
                    }
                }
            }

            float* c_row = C + (uint64_t)i * r;
            for (int j = 0; j < r; j++) {
                c_row[j] = (float)acc[j] * dequant;
            }
        }
    }
    return 0;
}

/* Quantize src_f32 -> dst_i8, using calib_scale if > 0 (static),
 * or computing scale dynamically from the data if calib_scale == 0. */
float tk_tf_quantize(struct tk_tensor* src_f32, struct tk_tensor* dst_i8,
                     float calib_scale) {
    if (calib_scale > 0.0f) {
        tk_ops_static_quantize_f32_i8(src_f32, dst_i8, calib_scale);
        return calib_scale;
    }
    return tk_ops_dyn_quantize_f32_i8(src_f32, dst_i8);
}
