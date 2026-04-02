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
    
    // set some reasonable value if not setted in config
    conv->input_c = 1;
    conv->has_bias  = 0;
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
    conv->filters = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, conv->dtype, (int[]){conv->num_filter, conv->input_c, conv->kernel_h, conv->kernel_w}, 4);
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

// called after conv_bind_input
struct tk_tensor* tk_conv_forward(struct tk_rt_ctx* ctx, struct tk_conv2d* conv, struct Dataset* dataset) {

    enum tk_dtype dtype = TK_F64;
    struct tk_tensor* padded_pics = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, (int[]){conv->num_filter, conv->padded_h, conv->padded_w}, 3);
    /* runtime tensor */
    struct tk_tensor* filtered_pics = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, (int[]){conv->num_filter, conv->filtered_h, conv->filtered_w}, 3); 

    if (ctx->rt_type != RT_DRYRUN) {
        for ( int i = 0 ; i < conv->num_filter ; ++i ) {
            tk_tensor_padding(dataset->samples, padded_pics,
                              conv->padding_h, conv->padding_w);
            tk_ops_convolute(padded_pics, conv->filters, filtered_pics);
            // tk_tensor_relu(filtered_pics);
        }
    }
    return filtered_pics;
}

