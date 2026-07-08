#ifndef TK_TENSOR_CHECK_H
#define TK_TENSOR_CHECK_H

#include "tensor.h"   // struct tk_tensor

int tk_check_shape_equal(struct tk_tensor* src1, struct tk_tensor* src2);
int tk_check_shape_equal_n(struct tk_tensor* src1, struct tk_tensor* src2, size_t n);
int tk_check_shape_equal_batch(struct tk_tensor* src1, struct tk_tensor* src2);
int tk_check_shape_equal_mult(struct tk_tensor* tensor_arr, uint32_t size);
int tk_check_contiguous_all(struct tk_tensor** tensors, int n, const char* func);
int tk_check_vec_matches_last_dim(struct tk_tensor* base, struct tk_tensor* vec,
                                   const char* vec_name, const char* func);
int tk_check_gemm_shape(struct tk_tensor* src1, struct tk_tensor* src2,
                            struct tk_tensor* dest,
                            int* out_p, int* out_q, int* out_r);

#endif
