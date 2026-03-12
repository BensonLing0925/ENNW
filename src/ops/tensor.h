#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include "arena.h"

#define TK_DISPATCH_TYPES(dtype, name, ...) \
    do {
        switch(dtype) { \
            case TK_F64: { typedef double scalar_t; __VA_ARGS__ break; } \
            case TK_F32: { typedef float  scalar_t; __VA_ARGS__ break; } \
            case TK_I16:  { typedef int16_t scalar_t; __VA_ARGS__ break; } \
            case TK_I8:  { typedef int8_t scalar_t; __VA_ARGS__ break; } \
            default: printf("Unsupported type in %s", name); \
        }
    } while (0)

enum tk_dtype {
    TK_NONE,
    TK_F64,
    TK_F32,
    TK_I16,
    TK_I8
};

struct tk_tensor {
    enum tk_dtype dtype;
    void* data;
    int ndims;
    int* shape;
    int* strides;
};

struct tk_tensor* tk_tensor_alloc(struct arena* a,
                                  enum tk_dtype dtype,
                                  int* shape,
                                  int ndims);

struct tk_tensor* tk_tensor_reshape(struct arena* a, struct tk_tensor* src, int* new_shape, int new_ndims);

int tk_tensor_transpose(struct arena* a, struct tk_tensor* tk_tensor, int dim1, int dim2);


#endif
