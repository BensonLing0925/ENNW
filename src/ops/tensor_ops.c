#include <omp.h>
#include <stdint.h>
#include "rt_error.h"
#include "float.h"
#include "math.h"
#include "tensor.h"
#include "tensor_check.h"
#include "pooling.h"
#include "tk_profiler.h"

// caller should allocate the correct dest data shape and space
int tk_ops_add(struct tk_tensor* src1, struct tk_tensor* src2,
               struct tk_tensor* dest) {

    // fast path: broadcast 1-D bias [N] onto [..., N]
    if (src2->ndims == 1 && src1->ndims >= 1 &&
        src1->shape[src1->ndims - 1] == src2->shape[0]) {

        int err = tk_check_shape_equal(src1, dest);
        if (err != 0) return err;

        if (!tk_tensor_is_contiguous(src1) || !tk_tensor_is_contiguous(src2) ||
            !tk_tensor_is_contiguous(dest))
            RT_FAIL(RT_EINVAL, "Incontiguous tensor detected\n");

        uint64_t rows = shape_size_calc(src1->shape, src1->ndims - 1);
        int      cols = src2->shape[0];

        TK_DISPATCH_TYPES(dest->dtype, __func__, {
            scalar_t* s1   = src1->data;
            scalar_t* bias = src2->data;
            scalar_t* d    = dest->data;
            for (uint64_t r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    d[r * cols + c] = s1[r * cols + c] + bias[c];
        });
        return 0;
    }

    int err = tk_check_shape_equal(src1, src2);
    if (err != 0)
        return err;

    err = tk_check_shape_equal(src1, dest);
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


// src1(p, q) x src2(q, r)
int tk_ops_gemm(struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest) {
    
    // if multiple dimensions for matmul, eg. A = [1, 12, 128, 64], B = [1, 12, 64, 256]
    // check A and B's dimensions before the last 2 dimensions matches or able to broadcast
    int err = tk_check_shape_equal_batch(src1, src2);
    if (err != 0)
        return err;

    // check inner dimension between 2 dimensions
    int src1_p = src1->shape[src1->ndims-2];
    int src1_q = src1->shape[src1->ndims-1];
    int src2_q = src2->shape[src2->ndims-2];
    int src2_r = src2->shape[src2->ndims-1];

    size_t work = src1_p * src1_q * src2_r;

    // check inner dimension between 2 dimensions
    if (src1_q != src2_q)
        RT_FAIL(RT_EINVAL, "GEMM inner dimension shape mismatch. src1 inner dim: %d, src2 inner dim: %d\n", 
                           src1_q,
                           src2_q);

    // check dest dimensions
    if ((dest->shape[dest->ndims-2] != src1_p) || (dest->shape[dest->ndims-1] != src2_r))
        RT_FAIL(RT_EINVAL, "GEMM dest dimension shape mismatch with src1 and src2. src1 last 2 dims: (%d, %d), src2 last 2 dims: (%d, %d)\n", src1_p, src1_q, src2_q, src2_r);

    if (!tk_tensor_is_contiguous(src1) || !tk_tensor_is_contiguous(src2) || !tk_tensor_is_contiguous(dest))
        RT_FAIL(RT_EINVAL, "Incontiguous tensor detected\n");
    
    uint64_t num_batches = 1;
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
            /*
            for ( int p = 0 ; p < src1->shape[src1->ndims-2] ; ++p )
                for ( int r = 0 ; r < src2->shape[src2->ndims-1] ; ++r ) {
                    scalar_t sum = 0;
                    for ( int q = 0 ; q < src1_q ; ++q ) {
                        sum += src1_ptr[p * src1_q + q] *
                               // src2_ptr read column, tend to miss cache
                               // maybe consider transpose it to dramatically increate cache hit
                               src2_ptr[q * src2_r + r];
                    } 
                    dest_ptr[p * src2_r + r] = sum;
                }
            */

            /* cache friendly version of naive gemm */
            /*
            for ( int p = 0 ; p < src1_p ; ++p )
                for ( int q = 0 ; q < src1_q ; ++q ) {
                    scalar_t val = src1_ptr[p * src1_q + q];
                    for ( int r = 0 ; r < src2_r ; ++r ) {
                        dest_ptr[p * src2_r + r] += val * src2_ptr[q * src2_r + r];
                    } 
                }
            */

            /* TILING gemm */
            /*
            int tile_q = 32;
            int tile_r = 32;
            // tile gemm is accumulated, clear zero first
            memset(dest_ptr, 0, src1_p * src2_r * sizeof(scalar_t));

            for ( int qq = 0 ; qq < src1_q ; qq += tile_q ) {
                int q_limit = (qq + tile_q > src1_q) ? src1_q : (qq + tile_q);
                for ( int rr = 0 ; rr < src2_r ; rr += tile_r ) {
                    int r_limit = (rr + tile_r > src2_r) ? src2_r : (rr + tile_r);
                    _Pragma("omp parallel for")
                    for ( int p = 0 ; p < src1_p ; ++p ) {
                        for ( int q = qq ; q < q_limit ; ++q ) {
                            scalar_t val = src1_ptr[p * src1_q + q];

                            for ( int r = rr ; r < r_limit ; ++r ) {
                                dest_ptr[p * src2_r + r] += val * src2_ptr[q * src2_r + r];
                            }
                        }
                    }
                    
                }
            }
            */
            int tile_q = 32;
            int tile_r = 32;

            _Pragma("omp parallel for")
            for (int p = 0; p < src1_p; ++p) {
                scalar_t* dest_row = dest_ptr + p * src2_r;
                memset(dest_row, 0, src2_r * sizeof(scalar_t));

                for (int qq = 0; qq < src1_q; qq += tile_q) {
                    int q_limit = (qq + tile_q > src1_q) ? src1_q : (qq + tile_q);
                    
                    for (int rr = 0; rr < src2_r; rr += tile_r) {
                        int r_limit = (rr + tile_r > src2_r) ? src2_r : (rr + tile_r);

                        for (int q = qq; q < q_limit; ++q) {
                            scalar_t val = src1_ptr[p * src1_q + q];
                            for (int r = rr; r < r_limit; ++r) {
                                dest_row[r] += val * src2_ptr[q * src2_r + r];
                            }
                        }
                    }
                }
            }
        }
    });
    return 0;
}

int tk_ops_layernorm(struct tk_tensor* src, struct tk_tensor* gamma, struct tk_tensor* beta, struct tk_tensor* dest) {

    // double eps = 1e-12;
	double eps = 1e-5; // gpt2

    // src and dest must have identical shape
    int err = tk_check_shape_equal(src, dest);
    if (err != 0)
        return err;

    // gamma and beta must match the last dimension of src
    int last_dim = src->shape[src->ndims - 1];
    if (gamma->shape[gamma->ndims - 1] != last_dim)
        RT_FAIL(RT_EINVAL, "Layernorm: gamma last dim %d != src last dim %d\n",
                gamma->shape[gamma->ndims - 1], last_dim);
    if (beta->shape[beta->ndims - 1] != last_dim)
        RT_FAIL(RT_EINVAL, "Layernorm: beta last dim %d != src last dim %d\n",
                beta->shape[beta->ndims - 1], last_dim);

    if (!tk_tensor_is_contiguous(src) || !tk_tensor_is_contiguous(gamma) ||
        !tk_tensor_is_contiguous(beta) || !tk_tensor_is_contiguous(dest))
        RT_FAIL(RT_EINVAL, "Incontiguous tensor detected\n");
    
    uint64_t num_rows = shape_size_calc(src->shape, src->ndims - 1);
    int dim = src->shape[src->ndims - 1];

    TK_DISPATCH_TYPES(src->dtype, __func__, {
        scalar_t* s_ptr = (scalar_t*)src->data;
        scalar_t* d_ptr = (scalar_t*)dest->data;
        scalar_t* g_ptr = (scalar_t*)gamma->data;
        scalar_t* b_ptr = (scalar_t*)beta->data;

        for (uint64_t i = 0; i < num_rows; i++) {
            scalar_t* row_s = s_ptr + i * dim;
            scalar_t* row_d = d_ptr + i * dim;

            /* mean */
            double sum = 0.0;
            for (int j = 0; j < dim; j++) sum += (double)row_s[j];
            double mean = sum / dim;

            /* variance */
            double var_sum = 0.0;
            for (int j = 0; j < dim; j++) {
                double diff = (double)row_s[j] - mean;
                var_sum += diff * diff;
            }
            double inv_std = 1.0 / sqrt(var_sum / dim + eps);

            /* normalize */
            for (int j = 0; j < dim; j++) {
                row_d[j] = (scalar_t)(((double)row_s[j] - mean) * inv_std
                                      * (double)g_ptr[j] + (double)b_ptr[j]);
            }
        }
    });
    return 0;
}

/* ---- fused add and layernorm --- */
int tk_ops_fused_add_norm(struct tk_tensor* x, struct tk_tensor* residual,
                          struct tk_tensor* gamma, struct tk_tensor* beta,
                          struct tk_tensor* out) {

    uint64_t num_rows = shape_size_calc(out->shape, out->ndims - 1);
    int dim = out->shape[out->ndims-1];
    // double eps = 1e-12;
    double eps = 1e-5;
    TK_DISPATCH_TYPES(x->dtype, __func__, {
        scalar_t* x_ptr = (scalar_t*)x->data;
        scalar_t* r_ptr = (scalar_t*)residual->data;
        scalar_t* g_ptr = (scalar_t*)gamma->data;
        scalar_t* b_ptr = (scalar_t*)beta->data;
        scalar_t* o_ptr = (scalar_t*)out->data;
        for ( uint64_t i = 0 ; i < num_rows ; ++i ) {
            scalar_t* row_x = x_ptr + i * dim;
            scalar_t* row_r = r_ptr + i * dim;
            scalar_t* row_o = o_ptr + i * dim;
            double sum = 0.0;
            double sum_sq = 0.0;
            for ( int j = 0 ; j < dim ; ++j ) {
                row_o[j] = row_x[j] + row_r[j]; 
                double val = row_o[j];
                sum += val;
                sum_sq += val * val;
            }
            double mean = sum / dim;
            double var = (sum_sq / dim) - (mean * mean);
            if (var < 0) var = 0.0;
            double inv_std = 1.0 / sqrt(var + eps);

            for ( int k = 0 ; k < dim ; k++ ) {
                row_o[k] = (scalar_t)((row_o[k] - mean) * inv_std * g_ptr[k] + b_ptr[k]);
            }

        }

    });
	return 0;
}

int tk_ops_fused_gemm_bias_gelu(struct tk_tensor* src1, struct tk_tensor* src2,
                          		struct tk_tensor* bias, struct tk_tensor* dest,
								int (*_gelu_fn) (void* src_data, void* dest_data, enum tk_dtype dtype, size_t size)) {

    // if multiple dimensions for matmul, eg. A = [1, 12, 128, 64], B = [1, 12, 64, 256]
    // check A and B's dimensions before the last 2 dimensions matches or able to broadcast
    int err = tk_check_shape_equal_batch(src1, src2);
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
        RT_FAIL(RT_EINVAL, "GEMM dest dimension shape mismatch with src1 and src2. src1 last 2 dims: (%d, %d), src2 last 2 dims: (%d, %d)\n", src1_p, src1_q, src2_q, src2_r);

    if (!tk_tensor_is_contiguous(src1) || !tk_tensor_is_contiguous(src2) || !tk_tensor_is_contiguous(dest))
        RT_FAIL(RT_EINVAL, "Incontiguous tensor detected\n");
    
    uint64_t num_batches = 1;
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
        scalar_t* bias_ptr = bias->data;

        // outer batch loop
        for (uint64_t b = 0; b < num_batches; ++b) {
            // locate to the current batch's base
            scalar_t* src1_ptr = src1_base + b * batch_stride_s1;
            scalar_t* src2_ptr = src2_base + b * batch_stride_s2;
            scalar_t* dest_ptr  = dest_base  + b * batch_stride_dest;

            int tile_q = 32;
            int tile_r = 32;
            _Pragma("omp parallel for")
            for (int p = 0; p < src1_p; ++p) {
                scalar_t* dest_row = dest_ptr + p * src2_r;
                memset(dest_row, 0, src2_r * sizeof(scalar_t));

                for (int qq = 0; qq < src1_q; qq += tile_q) {
                    int q_limit = (qq + tile_q > src1_q) ? src1_q : (qq + tile_q);
                    
                    for (int rr = 0; rr < src2_r; rr += tile_r) {
                        int r_limit = (rr + tile_r > src2_r) ? src2_r : (rr + tile_r);

                        for (int q = qq; q < q_limit; ++q) {
                            scalar_t val = src1_ptr[p * src1_q + q];
                            for (int r = rr; r < r_limit; ++r) {
                                dest_row[r] += val * src2_ptr[q * src2_r + r];
                            }
                        }
                    }
                }
                
                // add bias immediately to achieve cache locality 
                // bias need to be broadcasted (only one row)
                for ( int i = 0 ; i < src2_r ; ++i ) {
                    dest_row[i] += bias_ptr[i];
                }

                // GELU takes a number and output a number
				/*
                for ( int i = 0 ; i < src2_r ; ++i ) {
                        double x = (double)dest_row[i];
                        dest_row[i] = (scalar_t)(0.5 * x * (1.0 + erf(x * 0.7071067811865476)));
                }
				*/
				_gelu_fn((void*)dest_row, (void*)dest_row, dest->dtype, src2_r);
				// printf("[fused] gelu_fn ptr = %p\n", (void*)_gelu_fn);
            }
        }
    });
    return 0;
}

/* Because of fused operation, we need a raw pointer version of both gelu */
int _tk_ops_gelu_erf(void*  src_data, void* dest_data, enum tk_dtype dtype, size_t size) {
	TK_DISPATCH_TYPES(dtype, __func__, {
        scalar_t* s_ptr = (scalar_t*)src_data;
        scalar_t* d_ptr = (scalar_t*)dest_data;

        for (uint64_t n = 0; n < size; ++n) {
            double x = (double)s_ptr[n];
            d_ptr[n] = (scalar_t)(0.5 * x * (1.0 + erf(x * 0.7071067811865476)));
        }

	});
	return 0;
}

int _tk_ops_gelu_tanh(void* src_data, void* dest_data, enum tk_dtype dtype, size_t size) {
	TK_DISPATCH_TYPES(dtype, __func__, {
        scalar_t* s_ptr = (scalar_t*)src_data;
        scalar_t* d_ptr = (scalar_t*)dest_data;

        // GeLU(x) formula: 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const double sqrt_2_over_pi = 0.7978845608028654;
        const double coeff = 0.044715;

        for (uint64_t n = 0; n < size; ++n) {
            double x = (double)s_ptr[n];
            double x_cube = x * x * x;
            double inner = sqrt_2_over_pi * (x + coeff * x_cube);
            d_ptr[n] = (scalar_t)(0.5 * x * (1.0 + tanh(inner)));
        }
	});
	return 0;
}


int tk_ops_gelu_erf(struct tk_tensor* src, struct tk_tensor* dest) {
    uint64_t total_size = shape_size_calc(dest->shape, dest->ndims);
	_tk_ops_gelu_erf((void*)src->data, (void*)dest->data, dest->dtype, total_size);
    return 0;
}

int tk_ops_gelu_tanh(struct tk_tensor* src, struct tk_tensor* dest) {
    uint64_t total_size = shape_size_calc(dest->shape, dest->ndims);
	_tk_ops_gelu_tanh((void*)src->data, (void*)dest->data, dest->dtype, total_size);
    return 0;
}

int tk_ops_scale(struct tk_tensor* tensor, double scale) {
    uint64_t size = shape_size_calc(tensor->shape, tensor->ndims);
    
    TK_DISPATCH_TYPES(tensor->dtype, __func__, {
        scalar_t* data = (scalar_t*)tensor->data;
        scalar_t s = (scalar_t)scale;

        // #pragma omp parallel for
        for (uint64_t i = 0; i < size; i++) {
            data[i] *= s;
        }
    });
    return 0;
}

int tk_ops_softmax(struct tk_tensor* src, struct tk_tensor* dest) {

    int err = tk_check_shape_equal(src, dest);
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

void _tk_ops_convolute_2d(struct tk_tensor* pic, 
                         struct tk_tensor* filter,
                         struct tk_tensor* out) {

    // Fetch the lowest two dimensions to do convolution
    int pic_h = pic->shape[pic->ndims - 2];
    int pic_w = pic->shape[pic->ndims - 1];

    int f_h = filter->shape[filter->ndims - 2];
    int f_w = filter->shape[filter->ndims - 1];

    int out_h = pic_h - f_h + 1;
    int out_w = pic_w - f_w + 1;

    TK_DISPATCH_TYPES(pic->dtype, "_tk_ops_convolute_2d", {
        scalar_t* pic_ptr    = (scalar_t*)pic->data;
        scalar_t* filter_ptr = (scalar_t*)filter->data;
        scalar_t* out_ptr    = (scalar_t*)out->data;

        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                double sum = 0.0;
                for (int fy = 0; fy < f_h; ++fy) {
                    for (int fx = 0; fx < f_w; ++fx) {
                        int pic_idx = (y + fy) * pic_w + (x + fx);
                        int f_idx   = fy * f_w + fx;
                        sum += (double)pic_ptr[pic_idx] * (double)filter_ptr[f_idx];
                    }
                }
                out_ptr[y * out_w + x] = (scalar_t)sum;
            }
        }
    });
}

int tk_ops_convolute(struct tk_tensor* pic, struct tk_tensor* filter, struct tk_tensor* out) {
    int planes = 1;
    // how many picture should we convolute
    for (int i = 0; i < pic->ndims - 2; ++i) {
        planes *= pic->shape[i];
    }

    int pic_plane_size = pic->shape[pic->ndims-2] * pic->shape[pic->ndims-1];
    int out_plane_size = out->shape[out->ndims-2] * out->shape[out->ndims-1];

    for (int p = 0; p < planes; ++p) {
        struct tk_tensor pic_view = *pic;
        pic_view.data = pic->data + (p * pic_plane_size);
        
        struct tk_tensor out_view = *out;
        out_view.data = out->data + (p * out_plane_size);

        _tk_ops_convolute_2d(&pic_view, filter, &out_view);
    }
    return 0;
}

int tk_ops_onehot(struct tk_tensor* labels, struct tk_tensor* out) {

    if (labels->ndims != 1 || out->ndims != 2)
        RT_FAIL(RT_EINVAL, "Tensor dimensions input unexpected");
    
    double* out_ptr = (double*)out->data;
    uint8_t* label_ptr = (uint8_t*)labels->data;

    int num_samples = labels->shape[0];
    int num_classes = out->shape[1];

    // labels is [num_sample]
    // out is [num_sample, num_classes]
    memset(out_ptr, 0, num_samples * num_classes * sizeof(double));
    for (int i = 0; i < num_samples; ++i) {
        int label_val = (int)label_ptr[i];
        
        if (label_val >= 0 && label_val < num_classes) {
            out_ptr[i * num_classes + label_val] = 1.0;
        }
    }
    return 0;
}

int tk_ops_pooling(struct tk_pooling_params* pooling, struct tk_tensor* src, struct tk_tensor* dest) {

    int input_c = pooling->input_c;
    int input_h = pooling->input_h;
    int input_w = pooling->input_w;

    int size_h = pooling->kernel_h;
    int size_w = pooling->kernel_w;

    int stride_h = pooling->stride_h;
    int stride_w = pooling->stride_w;

    int pooled_h = pooling->pooled_h;
    int pooled_w = pooling->pooled_w;

    TK_DISPATCH_TYPES(src->dtype, "tk_ops_pooling", {
        // Inside this block, 'scalar_t' is defined as the correct type
        scalar_t* src_ptr = (scalar_t*)src->data;
        scalar_t* dest_ptr = (scalar_t*)dest->data;

        for (int c = 0; c < input_c; c++) {
            for (int ph = 0; ph < pooled_h; ph++) {
                for (int pw = 0; pw < pooled_w; pw++) {
                    
                    double biggest = -DBL_MAX;
                    for (int f_y = 0; f_y < size_h; f_y++) {
                        for (int f_x = 0; f_x < size_w; f_x++) {
                            int src_y = ph * stride_h + f_y;
                            int src_x = pw * stride_w + f_x;
                            
                            // Bounds check (optional but recommended)
                            if (src_y < input_h && src_x < input_w) {
                                double val = (double)src_ptr[c * input_h * input_w + src_y * input_w + src_x];
                                if (val > biggest) biggest = val;
                            }
                        }
                    }
                    dest_ptr[c * pooled_h * pooled_w + ph * pooled_w + pw] = (scalar_t)biggest;
                }
            }
        }
    });
    return 0;
}

/* embedding related operations */
int tk_ops_embedding_lookup(struct tk_tensor* input, struct tk_tensor* emb_weights, struct tk_tensor* output) {
    
    // check if the first(only) dim is the same as output's first dim
    if (tk_check_shape_equal_n(input, output, 1)) return -1;
    
    if (output->ndims < 2)
        RT_FAIL(RT_EINVAL, "Output tensor should at least have two dimension\n");

    // emb weight requirements
    if (emb_weights->ndims < 2)
        RT_FAIL(RT_EINVAL, "Embedding tensor should at least have two dimension\n");

    if (output->shape[1] != emb_weights->shape[1])
        RT_FAIL(RT_EINVAL, "Dimension mismatch between output and embedding tensor, output->shape[1]: %d, emb_weights->shape[1]: %d\n",
                            output->shape[1], emb_weights->shape[1]);

    if (emb_weights->dtype != output->dtype)
        RT_FAIL(RT_EINVAL, "Data type mismatch between output and emb_weight, output->dtype: %s, emb_weights->dtype: %s\n", tk_tensor_dtype_to_str(output->dtype), tk_tensor_dtype_to_str(emb_weights->dtype));
    
    // naive lookup using index
    int seq_len = input->shape[0];
    int hidden_dim = emb_weights->shape[1];
    TK_DISPATCH_TYPES(output->dtype, __func__, {
        int* input_ptr = (int*)input->data;
        scalar_t* src_ptr = (scalar_t*)emb_weights->data;
        scalar_t* dest_ptr = (scalar_t*)output->data;
        for (int i = 0 ; i < seq_len ; ++i) {
            int idx = input_ptr[i];
            if (idx < 0 || idx >= emb_weights->shape[0])
                RT_FAIL(RT_EINVAL, "Index out of range, idx: %d, emb_weights shape[0]: %d\n",
                        idx, emb_weights->shape[0]);
            memcpy(dest_ptr + (i * hidden_dim), src_ptr + (idx * hidden_dim), hidden_dim * sizeof(scalar_t));
        }
    });
    return 0;
}

/* decoder-specific related operations */
