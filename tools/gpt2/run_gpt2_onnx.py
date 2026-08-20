"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.exporters.onnx import onnx_export_from_model

m = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
m.config._attn_implementation = "eager"
AutoTokenizer.from_pretrained("gpt2").save_pretrained("gpt2_onnx/eager")
onnx_export_from_model(m, output="gpt2_onnx/eager", task="text-generation-with-past")
"""

import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = "gpt2_onnx/runtime_dump.onnx"
so.intra_op_num_threads = 4
ort.InferenceSession("gpt2_onnx/eager_opt/model.onnx", so,
                     providers=["CPUExecutionProvider"])
