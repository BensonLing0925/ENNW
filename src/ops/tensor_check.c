#include "../error/rt_error.h"
#include "tensor_check.h"

int tk_check_shape_equal(struct tk_tensor* src1, struct tk_tensor* src2) {
    if (src1->ndims != src2->ndims)
        RT_FAIL(RT_EINVAL, "Number of shape dimensions mismatch. src1: %d, src2: %d\n", src1->ndims, src2->ndims);

    for ( int i = 0 ; i < src1->ndims ; ++i )
        if (src1->shape[i] != src2->shape[i])
            RT_FAIL(RT_EINVAL, "Shape mismatch at index %d, with src1: %d and src2: %d", i, src1->shape[i], src2->shape[i]);

    return 0;
}

int tk_check_shape_equal_n(struct tk_tensor* src1, struct tk_tensor* src2, size_t n) {
    if (n > (size_t)src1->ndims || n > (size_t)src2->ndims)
        RT_FAIL(RT_EINVAL, "Number of dimension too large: %zu\n", n);
    for ( size_t i = 0 ; i < n ; ++i )
        if (src1->shape[i] != src2->shape[i])
            RT_FAIL(RT_EINVAL, "Shape mismatch at index %d, with src1: %d and src2: %d", i, src1->shape[i], src2->shape[i]);
    return 0;
}


int tk_check_shape_equal_batch(struct tk_tensor* src1, struct tk_tensor* src2) {

    if (src1->ndims != src2->ndims)
        RT_FAIL(RT_EINVAL, "Number of shape dimensions mismatch. src1: %d, src2: %d\n", src1->ndims, src2->ndims);

    for ( int i = 0 ; i < src1->ndims-2 ; ++i )
        if (src1->shape[i] != src2->shape[i])
            RT_FAIL(RT_EINVAL, "Shape mismatch at index %d, with src1: %d and src2: %d", i, src1->shape[i], src2->shape[i]);

    return 0;
}

int tk_check_shape_equal_mult(struct tk_tensor* tensor_arr, uint32_t size) {
    int err = 0;
    for ( uint32_t i = 0 ; i < size-1 ; ++i ) {
        err = tk_check_shape_equal(&tensor_arr[i], &tensor_arr[i+1]);
        if (err != 0)
            return err;
    }
    return 0;
}

int tk_check_contiguous_all(struct tk_tensor** tensors, int n, const char* func) {
    for (int i = 0; i < n; ++i)
        if (!tk_tensor_is_contiguous(tensors[i]))
            RT_FAIL(RT_EINVAL, "%s: tensor at index %d is not contiguous\n", func, i);
    return 0;
}

int tk_check_vec_matches_last_dim(struct tk_tensor* base, struct tk_tensor* vec,
                                                  const char* vec_name, const char* func) {
    int last_dim = base->shape[base->ndims - 1];
    if (vec->ndims != 1 || vec->shape[0] != last_dim)
        RT_FAIL(RT_EINVAL, "%s: %s shape mismatch, expected [%d], got ndims=%d shape[0]=%d\n",
                func, vec_name, last_dim, vec->ndims, vec->ndims > 0 ? vec->shape[0] : -1);
    return 0;
}

int tk_check_gemm_shape(struct tk_tensor* src1, struct tk_tensor* src2,
                            struct tk_tensor* dest,
                            int* out_p, int* out_q, int* out_r) {
    RT_CHECK(tk_check_shape_equal_batch(src1, src2));

    int p = src1->shape[src1->ndims-2];
    int q = src1->shape[src1->ndims-1];
    int q2 = src2->shape[src2->ndims-2];
    int r = src2->shape[src2->ndims-1];

    if (q != q2)
        RT_FAIL(RT_EINVAL, "GEMM inner dimension mismatch: %d vs %d\n", q, q2);
    if (dest->shape[dest->ndims-2] != p || dest->shape[dest->ndims-1] != r)
        RT_FAIL(RT_EINVAL, "GEMM dest shape mismatch: expected (%d, %d)\n", p, r);

    *out_p = p; *out_q = q; *out_r = r;
    return 0;
}

int tk_check_weight_is_i8(struct tk_tensor* weight) {
    return weight->dtype == TK_I8;
}
