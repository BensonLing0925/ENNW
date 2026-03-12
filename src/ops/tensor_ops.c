#include "../error/rt_error.h"
#include "math.h"
#include "tensor.h"

static int mult_shape_equal_check(struct tk_tensor* tensor_arr, uint32_t size) {
    int err = 0;
    for ( uint32_t i = 0 ; i < size-1 ; ++i ) {
        err = shape_equal_check(tensor_arr[i], tensor_arr[i+1]);
        if (err != 0)
            return err;
    }
    return 0;
}

static int shape_equal_check(struct tk_tensor* src1, struct tk_tensor* src2) {
    if (src1->ndims != src2->ndims)
        RT_FAIL(RT_EINVAL, "Number of shape dimensions mismatch. src1: %d, src2: %d\n", src1->ndims, src2->ndims);

    for ( int i = 0 ; i < src1->ndims ; ++i )
        if (src1->shape[i] != src2->shape[i])
            RT_FAIL(RT_EINVAL, "Shape mismatch at index %d, with src1: %d and src2: %d", i, src1->shape[i], src2->shape[i]);

    return 0;
}

// caller should allocate the correct dest data shape and space
int tk_ops_add(struct tk_tensor* src1, struct tk_tensor* src2,
               struct tk_tensor* dest) {
    
    int err = shape_equal_check(src1, src2);
    if (err != 0)
        return err;

    err = shape_equal_check(src1, dest);
    if (err != 0) {
        return err;
    }

    // check if tensors are contiguous
    if (!tk_tensor_is_contiguous(src1) || !tk_tensor_is_contiguous(src2) || !tk_tensor_is_contiguous(dest))
        RT_FAIL(RT_EINVAL, "Incontiguous tensor detected\n");

    uint64_t total_size = shape_size_calc(dest->shape, dest->ndims);

    // dtype is determined by dest, so we can change data type easily
    TK_DISPATCH_TYPES(dest->dtype, __func__, {
            scalar_t* src1_data = src1->data;
            scalar_t* src2_data = src2->data;
            scalar_t* dest_data = dest->data;
            // this part can be optimized using omp for parallelism
            for ( uint64_t n = 0 ; n < total_size ; ++n )
                dest_data[n] = src1_data[n] + src2_data[n];
            });

    return 0;
}

static int batch_shape_equal_check(struct tk_tensor* src1, struct tk_tensor* src2) {

    if (src1->ndims != src2->ndims)
        RT_FAIL(RT_EINVAL, "Number of shape dimensions mismatch. src1: %d, src2: %d\n", src1->ndims, src2->ndims);

    for ( int i = 0 ; i < src1->ndims-2 ; ++i )
        if (src1->shape[i] != src2->shape[i])
            RT_FAIL(RT_EINVAL, "Shape mismatch at index %d, with src1: %d and src2: %d", i, src1->shape[i], src2->shape[i]);

    return 0;
}

// src1(p, q) x src2(q, r)
int tk_ops_gemm(struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest) {
    
    // if multiple dimensions for matmul, eg. A = [1, 12, 128, 64], B = [1, 12, 64, 256]
    // check A and B's dimensions before the last 2 dimensions matches or able to broadcast
    int err = batch_shape_equal_check(src1, src2);
    if (err != 0)
        return err;

    // check inner dimension between 2 dimensions
    int src1_p = src1->shape[src1->ndims-2];
    int src1_q = src1->shape[src1->ndims-1];
    int src2_q = src2->shape[src2->ndims-2];
    int src2_r = src2->shape[src2->ndims-1];
    // check inner dimension between 2 dimensions
    if (src1_q != src2_q)
        RT_FAIL(RT_EINVAL, "GEMM inner dimension shape mismatch. src1 inner dim: %d, src2 inner dim: %d\n", 
                           src1_q,
                           src2_q);

    // check dest dimensions
    if ((dest->shape[dest->ndims-2] != src1_p) || (dest->shape[dest->ndims-1] != src2_r))
        RT_FAIL(RT_EINVAL, "GEMM dest dimension shape mismatch with src1 and src2. src1 last 2 dims: (%d, %d), src2 last 2 dims: (%d, %d)\n", 
                           src1_p, src1_q
                           src2_q, src2_r);

    if (!tk_tensor_is_contiguous(src1) || !tk_tensor_is_contiguous(src2) || !tk_tensor_is_contiguous(dest))
        RT_FAIL(RT_EINVAL, "Incontiguous tensor detected\n");
    
    uint64_t batch_size = 1;
    for (int i = 0; i < src1->ndims - 2; ++i) {
        num_batches *= src1->shape[i];
    }

    uint64_t batch_stride_s1 = (num_batches > 1) ? (src1_p * src1_q) : 0;
    uint64_t batch_stride_s2 = (num_batches > 1) ? (src2_q * src2_r) : 0;
    uint64_t batch_stride_dest = src1_p * src2_r;

    // inner loop deal with 2 dimensions matmul
    TK_DISPATCH_TYPES(dest->dtype, __func__, {

        scalar_t* src1_base = src1->data;
        scalar_t* src2_base = src2->data;
        scalar_t* dest_base = dest->data;

        // outer batch loop
        for (uint64_t b = 0; b < num_batches; ++b) {
            // locate to the current batch's base
            scalar_t* src1_ptr = src1_base + b * batch_stride_s1;
            scalar_t* src2_ptr = src2_base + b * batch_stride_s2;
            scalar_t* dest_ptr  = dest_base  + b * batch_stride_dest;

            // this part can be optimized using omp for parallelism
            for ( int p = 0 ; p < src1->shape[ndims-2] ; ++p )
                for ( int r = 0 ; r < src2->shape[ndims-1] ; ++r ) {
                    scalar_t sum = 0;
                    for ( int q = 0 ; q < src1_q ; ++q ) {
                        sum += src1_ptr[p * src1_q + q] *
                               // src2_ptr read column, tend to miss cache
                               // maybe consider transpose it to dramatically increate cache hit
                               src2_ptr[q * src2_r + r];
                    } 
                    dest_ptr[p * src2_r + r] = sum;
                }
            });

    return 0;
}

int tk_ops_layernorm(struct tk_tensor* src, struct tk_tensor* gamma, struct tk_tensor* beta, struct tk_tensor* dest) {

    uint32_t check_size = 4;
    struct tk_tensor* tensor_arr = malloc(check_size * sizeof(struct tk_tensor));

    tensor_arr[0] = src;
    tensor_arr[1] = gamma;
    tensor_arr[2] = beta;
    tensor_arr[3] = dest;

    int err = 0;
    err = mult_shape_equal_check(tensor_arr, check_size);
    if (err != 0)
        return err;

    if (!tk_tensor_is_contiguous(src) || !tk_tensor_is_contiguous(gemma) || 
        !tk_tensor_is_contiguous(beta) || !tk_tensor_is_contiguous(dest))
        RT_FAIL(RT_EINVAL, "Incontiguous tensor detected\n");
    
    uint64_t num_rows = shape_size_calc(src->shape, src->ndims - 1); // how many column to compute for layernorm
    int dim = src->shape[src->ndims - 1]; // last row's element count
    float eps = 1e-5f;

    TK_DISPATCH_TYPES(src->dtype, __func__, {
        scalar_t* s_ptr = (scalar_t*)src->data;
        scalar_t* d_ptr = (scalar_t*)dest->data;
        scalar_t* g_ptr = (scalar_t*)gamma->data;
        scalar_t* b_ptr = (scalar_t*)beta->data;

        // #pragma omp parallel for
        for (uint64_t i = 0; i < num_rows; i++) {
            scalar_t* row_s = s_ptr + i * dim;
            scalar_t* row_d = d_ptr + i * dim;

            // step 1. calculate arithmetic mean
            scalar_t sum = 0;
            // #pragma omp simd reduction(+:sum) // use simd to potentially speed up mean calculation
            for (int j = 0; j < dim; j++) sum += row_s[j];
            scalar_t mean = sum / dim;

            // step 2. calculate variance
            scalar_t var_sum = 0;
            // #pragma omp simd reduction(+:var_sum)
            for (int j = 0; j < dim; j++) {
                scalar_t diff = row_s[j] - mean;
                var_sum += diff * diff;
            }
            scalar_t inv_std = 1.0f / sqrtf(var_sum / dim + eps);

            // step 3. normalization
            for (int j = 0; j < dim; j++) {
                row_d[j] = (row_s[j] - mean) * inv_std * g_ptr[j] + b_ptr[j];
            }
        }
    });
    return 0;
}

void tk_ops_gelu(struct tk_tensor* src, struct tk_tensor* dest) {
    uint64_t total_size = shape_size_calc(dest->shape, dest->ndims);
    
    TK_DISPATCH_TYPES(dest->dtype, __func__, {
        scalar_t* s_ptr = (scalar_t*)src->data;
        scalar_t* d_ptr = (scalar_t*)dest->data;

        // GeLU(x) formula: 0.5x * (1 + tanh(sqrt(2 / pi)(0.044715 * x_cube)))
        const float sqrt_2_over_pi = 0.79788456f; // sqrt(2/pi)
        const float coeff = 0.044715f;

        // #pragma omp parallel for
        for (uint64_t n = 0; n < total_size; ++n) {
            float x = (float)s_ptr[n];
            float x_cube = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cube);
            d_ptr[n] = 0.5f * x * (1.0f + tanhf(inner));
        }
    });
}

int tk_ops_softmax(struct tk_tensor* src, struct tk_tensor* dest) {

    int err = shape_equal_check(src, dest);
    if (err != 0)
        return err;

    // take the last dimension of src
    int dim = src->shape[src->ndims-1];
    uint64_t num_rows = shape_size_calc(src->shape, src->ndims - 1);    // how many rows to calculate

    TK_DISPATCH_TYPES(dest->dtype, __func__, {
        scalar_t* s_data = (scalar_t*)src->data;
        scalar_t* d_data = (scalar_t*)dest->data;

        // #pragma omp parallel for
        for (uint64_t i = 0; i < num_rows; i++) {
            scalar_t* s_row = s_data + i * dim;
            scalar_t* d_row = d_data + i * dim;

            // 1. Find Max
            scalar_t max_val = s_row[0];
            for (int j = 1; j < dim; j++) {
                if (s_row[j] > max_val) max_val = s_row[j];
            }

            // 2. Exp and Sum
            double sum = 0; // use double to increase calc accuracy
            for (int j = 0; j < dim; j++) {
                d_row[j] = (scalar_t)exp((double)s_row[j] - (double)max_val);
                sum += (double)d_row[j];
            }

            // 3. Normalize
            for (int j = 0; j < dim; j++) {
                d_row[j] /= (scalar_t)sum;
            }
        }
    });

    return 0;
}
