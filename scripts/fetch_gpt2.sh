#!/bin/bash
# fetch_gpt2.sh
set -e

ENNW_HOME="$(cd "$(dirname "$0")/.." && pwd)"
GPT2_DIR="$ENNW_HOME/data/gpt2"
mkdir -p "$GPT2_DIR"
curl -L -o "$GPT2_DIR/gpt2_model.safetensors" \
	https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
