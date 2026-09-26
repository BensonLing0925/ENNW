#include "onnx.proto3.pb-c.h"
#include "onnx/importer.h"
#include "fileio.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("usage: ./executable model.onnx\n");
        return 0;
    }

    struct tk_file_data* r = tk_file_read_all(argv[1]);
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

