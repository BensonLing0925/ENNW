#include <omp.h>
#include <stdint.h>
#include "../error/rt_error.h"
#include "float.h"
#include "math.h"
#include "tensor.h"
#include "../modules/pooling/pooling.h"

/* input: a list of integer
 * output: a list of corresponding vectors
 * Caller pre-allocate output->data 
 */
/* ---- int8 dynamic quantization ---- */

float tk_ops_dyn_quantize_f32_i8(struct tk_tensor* src_f32, struct tk_tensor* dst_i8);
void tk_ops_static_quantize_f32_i8(struct tk_tensor* src_f32,
                                    struct tk_tensor* dst_i8,
                                    float scale);
int tk_ops_gemm_i8f32(struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dst);
float tk_tf_quantize(struct tk_tensor* src_f32, struct tk_tensor* dst_i8,
                     float calib_scale);
