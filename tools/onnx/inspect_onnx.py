import onnx
import os
from collections import Counter

path = os.path.expanduser(
    "~/ENNW/data/gpt2_q8_Xenova/model_int8.onnx"
)
model = onnx.load(path)

print("IR version:", model.ir_version)

print("\nOpsets:")
for x in model.opset_import:
    print(x.domain, x.version)

def count_graph_ops(graph, counts, depth=0):
    for node in graph.node:
        counts[node.op_type] += 1

        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                count_graph_ops(attr.g, counts, depth + 1)

            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    count_graph_ops(g, counts, depth + 1)

print("\nOperators:")
counts = Counter()
count_graph_ops(model.graph, counts)

for op, count in sorted(counts.items()):
    print(f"{op:30} {count}")


print("\nInitializers:")
types = Counter(t.data_type for t in model.graph.initializer)

for dtype, count in types.items():
    print(dtype, count)

for t in model.graph.initializer:
    print(
        t.name,
        onnx.TensorProto.DataType.Name(t.data_type),
        list(t.dims)
    )
