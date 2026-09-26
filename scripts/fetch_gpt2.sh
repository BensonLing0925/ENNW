#!/bin/bash
# fetch_gpt2.sh
set -e

ENNW_HOME="$(cd "$(dirname "$0")/.." && pwd)"
SELF="${0##*/}"
# FP32 GPT-2
GPT2_DIR="$ENNW_HOME/data/gpt2"
GPT2_MODEL="$GPT2_DIR/gpt2_model.safetensors"

if [ ! -f "$GPT2_MODEL" ]; then
    echo "[$SELF] Setup and Download GPT2 Model"
    mkdir -p "$GPT2_DIR"

    curl -fL -o "$GPT2_MODEL.tmp" \
        https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors

    mv "$GPT2_MODEL.tmp" "$GPT2_MODEL"
fi


# INT8 GPT-2 (Xenova / ONNX)
GPT2_Q8_DIR="$ENNW_HOME/data/gpt2_q8_Xenova"
GPT2_Q8_MODEL="$GPT2_Q8_DIR/model_int8.onnx"

if [ ! -f "$GPT2_Q8_MODEL" ]; then
    echo "[$SELF] Setup and Download GPT2 Q8 Xenova Model"
    mkdir -p "$GPT2_Q8_DIR"

    curl -fL -o "$GPT2_Q8_MODEL.tmp" \
        https://huggingface.co/Xenova/gpt2/resolve/main/onnx/model_int8.onnx

    mv "$GPT2_Q8_MODEL.tmp" "$GPT2_Q8_MODEL"
fi
