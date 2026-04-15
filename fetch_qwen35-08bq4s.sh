#!/bin/bash

wget https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_S.gguf

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3.5-0.8B Qwen3.5-0.8B-tokenizer

python3 gguf_to_weed.py --input Qwen3.5-0.8B-Q4_K_S.gguf --output Qwen3.5-0.8B-Q4_K_S.weed
