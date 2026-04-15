#include <stdio.h>
#include <stdlib.h>
#include "structDef.h"
#include "conv.h"
#include "../mem/arena.h"
#include "../../nn_utils/nn_utils.h"
#include "../../ops/tensor.h"
#include "../../ops/tensor_ops.h"
#include "../../runtime/rt_context.h"

struct tk_conv2d* tk_conv2D_create(struct tk_rt_ctx* ctx) {
    struct tk_conv2d* conv = arena_alloc(ctx->meta_arena, sizeof(struct tk_conv2d));
    return conv;
}

void tk_conv2d_init(struct tk_conv2d* conv, struct tk_conv2d_config config) {
    conv->num_filter = config.num_filter;
    conv->kernel_h   = config.kernel_h;
    conv->kernel_w   = config.kernel_w;
    conv->stride_h  = config.stride_h;
    conv->stride_w  = config.stride_w;
    conv->padding_h = config.padding_h;
    conv->padding_w = config.padding_w;

    conv->input_c  = 1;
    conv->has_bias = 0;
    conv->dtype    = TK_F64;   /* filters are always double */
}

// conv->input_h/w and kernel_w/h should be setted before calling this
// function
void tk_conv2d_setup(struct tk_conv2d* conv, struct tk_tensor* input) {

    // input information (should be 3D. If gray scale, the first dimension saet to 1)
    conv->input_c = input->shape[0];
    conv->input_h = input->shape[1];
    conv->input_w = input->shape[2];
    conv->dtype = input->dtype;

    // inferred information
    conv->filtered_h = (conv->input_h + 2 * conv->padding_h - conv->kernel_h) / conv->stride_h + 1;
    conv->filtered_w = (conv->input_w + 2 * conv->padding_w - conv->kernel_w) / conv->stride_w + 1;
    conv->padded_h = conv->input_h + 2 * conv->padding_h;
    conv->padded_w = conv->input_w + 2 * conv->padding_w;

}

int tk_conv2d_load_weights(struct tk_conv2d* conv, FILE* fp) {
    if (!fp)
        RT_FAIL(RT_EINVAL, "File pointer is NULL");
    uint64_t size = conv->input_c * conv->kernel_h * conv->kernel_w * tk_get_dtype_size(conv->dtype);
    int err = tk_tensor_load_data(conv->filters, fp, size);
    return err;
}

void tk_conv2d_alloc(struct tk_rt_ctx* ctx, tk_conv2d* conv) {
    int filter_shape[4] = {conv->num_filter, conv->input_c, conv->kernel_h, conv->kernel_w};
    /* Weights are persistent — allocate from data_arena, not the workspace */
    conv->filters = tk_tensor_alloc(ctx->data_arena, conv->dtype, filter_shape, 4);

    /* Xavier-style random init */
    uint64_t total = shape_size_calc(filter_shape, 4);
    double* data = (double*)conv->filters->data;
    for (uint64_t i = 0; i < total; ++i)
        data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.01;
}

/*
// called after initConv and allocConv
void manualKernal(tk_conv2d* conv) {

	int num_filter = conv->num_filter;
	double kernels[10][3][3] = {
			// Horizontal edge detection
			{
				{-1, -1, -1},
				{0, 0, 0},
				{1, 1, 1}
			},
			// Vertical edge detection
			{
				{-1, 0, 1},
				{-1, 0, 1},
				{-1, 0, 1}
			},
			// Diagonal edge detection (135)
			{
				{-1, -1, 0},
				{-1, 0, 1},
				{0, 1, 1}
			},
			// Diagonal edge detection (45)
			{
				{0, -1, -1},
				{1, 0, -1},
				{1, 1, 0}
			},
			// Sharpen Filter
			{
				{0, -1, 0},
				{-1, 5, -1},
				{0, -1, 0}
			},
			// Sobel Horizontal 
			{
				{-1, -2, -1},
				{0, 0, 0},
				{1, 2, 1}
			},
			// Sobel Vertical 
			{
				{-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1}
			},
			// Horizontal line detection
			{
				{1, 1, 1},
				{0, 0, 0},
				{-1, -1, -1}
			},
			// Vertical line detection
			{
				{1, 0, -1},
				{1, 0, -1},
				{1, 0, -1}
			},
			// 45 line detection
			{
				{-2, -1, 0},
				{-1, 0, 1},
				{0, 1, 2}
			}
	};

	for ( int filterCnt = 0 ; filterCnt < num_filter ; ++filterCnt ) {
		for ( int fRowSize = 0 ; fRowSize < conv->kernel_h ; ++fRowSize ) {  // square filter assumed
			for ( int fColSize = 0 ; fColSize < conv->kernel_w ; ++fColSize ) {  // square filter assumed
				conv->filters[filterCnt][fRowSize][fColSize] = kernels[filterCnt][fRowSize][fColSize];
			}		
		}		
	}		
}		
*/

/* called after tk_conv2d_setup + tk_conv2d_alloc
 *
 * Correct 2-D convolution:
 *   input:   [in_c, in_h, in_w]   (single sample)
 *   filters: [num_filter, in_c, kH, kW]
 *   output:  [num_filter, filtered_h, filtered_w]
 *
 * output[f, oh, ow] = sum_{c,kh,kw} input[c, oh*sh+kh, ow*sw+kw] * filter[f,c,kh,kw]
 */
struct tk_tensor* tk_conv_forward(struct tk_rt_ctx* ctx,
                                  struct tk_conv2d* conv,
                                  struct Dataset* dataset) {
    enum tk_dtype dtype = TK_F64;

    /* Allocate workspace only for the dry-run sizing pass */
    if (conv->padding_h > 0 || conv->padding_w > 0)
        tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype,
                           (int[]){conv->input_c, conv->padded_h, conv->padded_w}, 3);

    int oH = conv->filtered_h;
    int oW = conv->filtered_w;
    struct tk_tensor* filtered_pics = tk_ws_tensor_alloc(
        ctx->ws, ctx->meta_arena, dtype,
        (int[]){conv->num_filter, oH, oW}, 3);

    if (ctx->rt_type == RT_DRYRUN) return filtered_pics;

    int F   = conv->num_filter;
    int C   = conv->input_c;
    int kH  = conv->kernel_h;
    int kW  = conv->kernel_w;
    int sH  = conv->stride_h;
    int sW  = conv->stride_w;
    int pH  = conv->padding_h;
    int pW  = conv->padding_w;
    int inH = conv->input_h;
    int inW = conv->input_w;

    double* input_data  = (double*)dataset->samples->data;  /* [C, inH, inW] */
    double* filter_data = (double*)conv->filters->data;      /* [F, C, kH, kW] */
    double* out_data    = (double*)filtered_pics->data;       /* [F, oH, oW] */

    memset(out_data, 0, (size_t)F * oH * oW * sizeof(double));

    for (int f = 0; f < F; ++f) {
        double* out_f = out_data + (size_t)f * oH * oW;
        for (int c = 0; c < C; ++c) {
            double* in_c  = input_data  + (size_t)c * inH * inW;
            double* filt  = filter_data + ((size_t)f * C + c) * kH * kW;
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    double sum = 0.0;
                    for (int kh = 0; kh < kH; ++kh) {
                        int ih = oh * sH + kh - pH;
                        if (ih < 0 || ih >= inH) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            int iw = ow * sW + kw - pW;
                            if (iw < 0 || iw >= inW) continue;
                            sum += in_c[ih * inW + iw] * filt[kh * kW + kw];
                        }
                    }
                    out_f[oh * oW + ow] += sum;
                }
            }
        }
    }
    return filtered_pics;
}

