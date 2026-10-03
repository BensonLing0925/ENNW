#ifndef TK_ONNX_IMPORT_H
#define TK_ONNX_IMPORT_H

#include "rt_context.h"
#include "onnx/scope.h"

struct tk_onnx_importer {
    struct tk_rt_ctx* ctx;
    struct tk_onnx_scope* scope;
};

Onnx__ModelProto* tk_onnx_model_get(struct tk_file_data* f);
void tk_onnx_model_free(Onnx__ModelProto* model);
int tk_onnx_to_tensor(struct tk_rt_ctx* ctx, Onnx__TensorProto* tp, struct tk_tensor** out);
int tk_onnx_tensor_data_copy(const Onnx__TensorProto *t, struct tk_tensor* tensor);
int tk_onnx_node_import(struct tk_rt_ctx* ctx, Onnx__NodeProto* node, struct tk_onnx_scope* cur_scope);
int tk_onnx_graph_import(struct tk_rt_ctx* ctx, Onnx__GraphProto* g, struct tk_onnx_scope* parent);

#endif
