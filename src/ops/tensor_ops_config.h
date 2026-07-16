#ifndef TK_OPS_CONFIG_H
#define TK_OPS_CONFIG_H

#include "tensor.h"

enum tk_ops_gelu_variant {
	TK_OPS_GELU_ERF,
	TK_OPS_GELU_TANH,
};

struct tk_ops_config {

	/* gelu related ops */
	enum tk_ops_gelu_variant gelu_variant;
	int (*gelu_fn) (struct tk_tensor* src, struct tk_tensor* dest);
	int (*gelu_fn_raw) (struct tk_tensor* src, struct tk_tensor* dest,
						enum tk_dtype, size_t size);

};

#endif
