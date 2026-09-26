#include "arena.h"
#include "fileio.h"
#include "rt_context.h"
#include "tensor.h"
#include "onnx.proto3.pb-c.h"
#include <stdio.h>
#include <stdlib.h>

// call tk_onnx_model_free to free
Onnx__ModelProto* tk_onnx_model_get(struct tk_file_data* f) {
    if (!f) {
        fprintf(stderr, "Input is NULL\n");
        return NULL;
    }
    Onnx__ModelProto* model = onnx__model_proto__unpack(
            NULL, f->bytes, f->data);

    if (!model) {
        fprintf(stderr, "Failed to unpack ONNX model\n");
        return NULL;
    }
    return model;
}

void tk_onnx_model_free(Onnx__ModelProto* model) {
    onnx__model_proto__free_unpacked(model, NULL);
}

static enum tk_dtype tk_onnx_to_tk_dtype(int32_t onnx_dtype) {
    switch (onnx_dtype) {
        case 0:
            fprintf(stderr, "Undefined ONNX data type\n");
            break;
        case 1:
            return TK_F32;
        case 2:
            return TK_U8;
        case 3:
            return TK_I8;
        case 5:
            return TK_I16;
        case 6:
            return TK_I32;
        case 11:
            return TK_F64;
        default:
            fprintf(stderr, "Unsupported TK data type\n");
    }
}

// use specifically for debug and
// figuring out the field in which the tensor's data stored
void tk_onnx_tensor_probe(const Onnx__TensorProto *tp)
{
    printf("name: %s\n", tp->name);
    printf("dtype: %d\n", tp->data_type);

    printf("dims:");
    for (size_t i = 0; i < tp->n_dims; ++i)
        printf(" %ld", (long)tp->dims[i]);
    printf("\n");

    printf("raw_data: %zu bytes\n", tp->raw_data.len);

    printf("float_data:  %zu\n", tp->n_float_data);
    printf("int32_data:  %zu\n", tp->n_int32_data);
    printf("int64_data:  %zu\n", tp->n_int64_data);
    printf("double_data: %zu\n", tp->n_double_data);
    printf("uint64_data: %zu\n", tp->n_uint64_data);
    printf("string_data: %zu\n", tp->n_string_data);

    printf("external_data: %zu\n", tp->n_external_data);

    if (tp->segment) {
        printf("segment: [%ld, %ld)\n",
               (long)tp->segment->begin,
               (long)tp->segment->end);
    } else {
        printf("segment: none\n");
    }
}

// called when parsing initializers
// callee allocate tensors
// out only needs to be a valid pointer
int tk_onnx_to_tensor(struct tk_rt_ctx* ctx, Onnx__TensorProto* tp, struct tk_tensor** out) {
    struct tk_tensor* tensor = NULL;
    enum tk_dtype dtype = tk_onnx_to_tk_dtype(tp->data_type);
    RT_CHECK(tk_tensor_alloc(ctx->data_arena, dtype, tp->dims, tp->n_dims, out));
    *out = tensor;
    return 0;
}

/*
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("usage: ./executable model.onnx\n");
        return 0;
    }

    struct file_data* r = data_read(argv[1]);
    Onnx__ModelProto* model = onnx__model_proto__unpack(
            NULL, r->bytes, r->data);
    printf("ir_version: %ld\n", model->ir_version);

    for (size_t i = 0; i < model->n_opset_import; ++i) {
        Onnx__OperatorSetIdProto *opset = model->opset_import[i];

        printf("opset domain=%s version=%ld\n",
               opset->domain ? opset->domain : "",
               (long)opset->version);
    }

    printf("nodes: %zu\n", model->graph->n_node);
    printf("initializers: %zu\n", model->graph->n_initializer);

    printf("Printing Initializers...\n");
    Onnx__TensorProto** tensors = model->graph->initializer;
    for ( size_t t = 0 ; t < model->graph->n_initializer ; ++t ) {
        Onnx__TensorProto* tensor = model->graph->initializer[t];
        printf("tensor [%zu]: %s\n", t, tensor->name);
        printf("Data type: %d\n", tensor->data_type);
        printf("dims:");

        for (size_t i = 0; i < tensor->n_dims; ++i)
            printf(" %ld", (long)tensor->dims[i]);

        printf("\n");

    }


    for (size_t i = 0; i < model->graph->n_node; ++i) {
        Onnx__NodeProto *node = model->graph->node[i];

        printf("node[%zu]: %s\n",
               i,
               node->op_type ? node->op_type : "(null)");
    }

    Onnx__NodeProto *node = model->graph->node[0];
    printf("attributes: %zu\n", node->n_attribute);
    for (size_t i = 0; i < node->n_attribute; ++i) {
        Onnx__AttributeProto *attr = node->attribute[i];

        printf("attr[%zu]: name=%s type=%d\n",
               i,
               attr->name ? attr->name : "(null)",
               attr->type);
    }

    for (size_t i = 0; i < node->n_attribute; ++i) {
        Onnx__AttributeProto *attr = node->attribute[i];

        if (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH) {
            printf("%s:\n", attr->name);
            printf("  nodes: %zu\n", attr->g->n_node);
            printf("  initializers: %zu\n", attr->g->n_initializer);
        }

        Onnx__GraphProto *g = attr->g;
        for (size_t j = 0; j < g->n_node; ++j) {
            printf("  node[%zu]: %s. ",
                   j,
                   g->node[j]->op_type);
            Onnx__NodeProto* node = g->node[j];
            for (size_t input = 0 ; input < node->n_input ; ++input) {
                if (input == 0 && input == node->n_input-1)
                    printf("\n  Inputs: %s", node->input[input]);
                else if (input == 0)
                    printf("\n  Inputs: %s ", node->input[input]);
                else if (input == node->n_input-1)
                    printf("%s", node->input[input]);
                else printf("%s ", node->input[input]);
            }

            for (size_t output = 0 ; output < node->n_output ; ++output) {
                if (output == 0 && output == node->n_output-1)
                    printf("\n  Outputs: %s\n", node->output[output]);
                else if (output == 0)
                    printf("\n  Outputs: %s ", node->output[output]);
                else if (output == node->n_output-1)
                    printf("%s\n", node->output[output]);
                else printf("%s ", node->output[output]);
            }
            printf("  Number of attributes: %zu\n", node->n_attribute);
            if (node->n_attribute > 0) {
                for (size_t i = 0; i < node->n_attribute; ++i) {
                    Onnx__AttributeProto *a = node->attribute[i];

                    printf("  attr: %s type=%d i=%ld\n",
                           a->name,
                           a->type,
                           (long)a->i);
                }
            }
        }
    }

    onnx__model_proto__free_unpacked(model, NULL);

    free(r->data);
    free(r);
}
*/
